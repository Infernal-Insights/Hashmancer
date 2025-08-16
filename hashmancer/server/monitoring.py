"""
Comprehensive Monitoring System
Provides real-time monitoring, metrics collection, alerting, and health checks
"""

import asyncio
import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import json
from collections import deque, defaultdict
import statistics
from datetime import datetime, timedelta
import weakref

from .redis_pool import redis_sync_connection, redis_async_connection
from .websocket_manager import websocket_manager, broadcast_message

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    type: MetricType
    value: Union[int, float]
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "type": self.type.value,
            "value": self.value,
            "timestamp": self.timestamp,
            "labels": self.labels
        }


@dataclass
class Alert:
    """Alert notification."""
    id: str
    name: str
    severity: AlertSeverity
    message: str
    timestamp: float
    resolved: bool = False
    resolved_at: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


class MetricsCollector:
    """Collects and stores metrics."""
    
    def __init__(self, max_history: int = 1000):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, List[float]] = defaultdict(list)
        self.timers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.lock = threading.RLock()
    
    def record_counter(self, name: str, value: float = 1, labels: Optional[Dict[str, str]] = None):
        """Record a counter metric."""
        with self.lock:
            key = self._make_key(name, labels)
            self.counters[key] += value
            
            metric = Metric(
                name=name,
                type=MetricType.COUNTER,
                value=self.counters[key],
                timestamp=time.time(),
                labels=labels or {}
            )
            self.metrics[key].append(metric)
    
    def record_gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a gauge metric."""
        with self.lock:
            key = self._make_key(name, labels)
            self.gauges[key] = value
            
            metric = Metric(
                name=name,
                type=MetricType.GAUGE,
                value=value,
                timestamp=time.time(),
                labels=labels or {}
            )
            self.metrics[key].append(metric)
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram metric."""
        with self.lock:
            key = self._make_key(name, labels)
            self.histograms[key].append(value)
            
            # Keep only last 1000 values
            if len(self.histograms[key]) > 1000:
                self.histograms[key] = self.histograms[key][-1000:]
            
            metric = Metric(
                name=name,
                type=MetricType.HISTOGRAM,
                value=value,
                timestamp=time.time(),
                labels=labels or {}
            )
            self.metrics[key].append(metric)
    
    def start_timer(self, name: str, labels: Optional[Dict[str, str]] = None) -> float:
        """Start a timer and return start time."""
        return time.time()
    
    def end_timer(self, name: str, start_time: float, labels: Optional[Dict[str, str]] = None):
        """End a timer and record the duration."""
        duration = time.time() - start_time
        with self.lock:
            key = self._make_key(name, labels)
            self.timers[key].append(duration)
            
            metric = Metric(
                name=name,
                type=MetricType.TIMER,
                value=duration,
                timestamp=time.time(),
                labels=labels or {}
            )
            self.metrics[key].append(metric)
    
    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create a unique key for metric with labels."""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}[{label_str}]"
    
    def get_metric_summary(self, name: str, labels: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        key = self._make_key(name, labels)
        
        with self.lock:
            if key not in self.metrics:
                return {}
            
            metric_data = list(self.metrics[key])
            if not metric_data:
                return {}
            
            latest_metric = metric_data[-1]
            values = [m.value for m in metric_data]
            
            summary = {
                "name": name,
                "type": latest_metric.type.value,
                "labels": labels or {},
                "count": len(values),
                "latest_value": latest_metric.value,
                "latest_timestamp": latest_metric.timestamp,
            }
            
            if len(values) > 1:
                summary.update({
                    "min": min(values),
                    "max": max(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                })
                
                if len(values) > 1:
                    summary["std_dev"] = statistics.stdev(values)
            
            return summary
    
    def get_all_metrics(self) -> List[Dict[str, Any]]:
        """Get all current metrics."""
        with self.lock:
            results = []
            for key in self.metrics:
                if self.metrics[key]:
                    latest_metric = self.metrics[key][-1]
                    results.append(latest_metric.to_dict())
            return results


class SystemMonitor:
    """Monitors system resources."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.monitoring = False
        self.monitor_task: Optional[asyncio.Task] = None
        self.monitor_interval = 5  # seconds
    
    async def start_monitoring(self):
        """Start system monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("System monitoring started")
    
    async def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_task and not self.monitor_task.done():
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("System monitoring stopped")
    
    async def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.monitor_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def _collect_system_metrics(self):
        """Collect system metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=None)
            self.metrics.record_gauge("system_cpu_percent", cpu_percent)
            
            cpu_count = psutil.cpu_count()
            self.metrics.record_gauge("system_cpu_count", cpu_count)
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.metrics.record_gauge("system_memory_total", memory.total)
            self.metrics.record_gauge("system_memory_used", memory.used)
            self.metrics.record_gauge("system_memory_percent", memory.percent)
            self.metrics.record_gauge("system_memory_available", memory.available)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.metrics.record_gauge("system_disk_total", disk.total)
            self.metrics.record_gauge("system_disk_used", disk.used)
            self.metrics.record_gauge("system_disk_percent", disk.used / disk.total * 100)
            
            # Network metrics
            network = psutil.net_io_counters()
            self.metrics.record_counter("system_network_bytes_sent", network.bytes_sent)
            self.metrics.record_counter("system_network_bytes_recv", network.bytes_recv)
            self.metrics.record_counter("system_network_packets_sent", network.packets_sent)
            self.metrics.record_counter("system_network_packets_recv", network.packets_recv)
            
            # Process metrics
            process = psutil.Process()
            self.metrics.record_gauge("process_memory_rss", process.memory_info().rss)
            self.metrics.record_gauge("process_memory_vms", process.memory_info().vms)
            self.metrics.record_gauge("process_cpu_percent", process.cpu_percent())
            self.metrics.record_gauge("process_num_threads", process.num_threads())
            self.metrics.record_gauge("process_num_fds", process.num_fds())
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self):
        self.alerts: Dict[str, Alert] = {}
        self.alert_rules: List[Dict[str, Any]] = []
        self.callbacks: List[Callable[[Alert], None]] = []
        self.lock = threading.RLock()
    
    def add_alert_rule(self, name: str, condition: Callable[[Dict[str, Any]], bool],
                      severity: AlertSeverity, message: str, 
                      metadata: Optional[Dict[str, Any]] = None):
        """Add an alert rule."""
        rule = {
            "name": name,
            "condition": condition,
            "severity": severity,
            "message": message,
            "metadata": metadata or {}
        }
        self.alert_rules.append(rule)
    
    def add_callback(self, callback: Callable[[Alert], None]):
        """Add alert callback."""
        self.callbacks.append(callback)
    
    def check_alerts(self, metrics: Dict[str, Any]):
        """Check all alert rules against current metrics."""
        for rule in self.alert_rules:
            try:
                if rule["condition"](metrics):
                    self._trigger_alert(rule)
                else:
                    self._resolve_alert(rule["name"])
            except Exception as e:
                logger.error(f"Error checking alert rule {rule['name']}: {e}")
    
    def _trigger_alert(self, rule: Dict[str, Any]):
        """Trigger an alert."""
        alert_id = f"alert_{rule['name']}_{int(time.time())}"
        
        with self.lock:
            if rule["name"] in self.alerts and not self.alerts[rule["name"]].resolved:
                return  # Alert already active
            
            alert = Alert(
                id=alert_id,
                name=rule["name"],
                severity=rule["severity"],
                message=rule["message"],
                timestamp=time.time(),
                metadata=rule["metadata"]
            )
            
            self.alerts[rule["name"]] = alert
            
            logger.warning(f"Alert triggered: {alert.name} - {alert.message}")
            
            # Notify callbacks
            for callback in self.callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    logger.error(f"Error in alert callback: {e}")
    
    def _resolve_alert(self, name: str):
        """Resolve an alert."""
        with self.lock:
            if name in self.alerts and not self.alerts[name].resolved:
                self.alerts[name].resolved = True
                self.alerts[name].resolved_at = time.time()
                
                logger.info(f"Alert resolved: {name}")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self.lock:
            return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def get_all_alerts(self) -> List[Alert]:
        """Get all alerts."""
        with self.lock:
            return list(self.alerts.values())


class HealthChecker:
    """Performs health checks on various system components."""
    
    def __init__(self):
        self.checks: Dict[str, Callable[[], Dict[str, Any]]] = {}
        self.health_status: Dict[str, Dict[str, Any]] = {}
        self.lock = threading.RLock()
    
    def register_check(self, name: str, check_func: Callable[[], Dict[str, Any]]):
        """Register a health check."""
        self.checks[name] = check_func
    
    async def run_checks(self) -> Dict[str, Any]:
        """Run all health checks."""
        results = {}
        overall_healthy = True
        
        for name, check_func in self.checks.items():
            try:
                start_time = time.time()
                
                if asyncio.iscoroutinefunction(check_func):
                    result = await check_func()
                else:
                    result = check_func()
                
                duration = time.time() - start_time
                
                check_result = {
                    "healthy": result.get("healthy", True),
                    "message": result.get("message", "OK"),
                    "details": result.get("details", {}),
                    "duration_ms": round(duration * 1000, 2),
                    "timestamp": time.time()
                }
                
                if not check_result["healthy"]:
                    overall_healthy = False
                
                results[name] = check_result
                
                with self.lock:
                    self.health_status[name] = check_result
                
            except Exception as e:
                error_result = {
                    "healthy": False,
                    "message": f"Health check failed: {e}",
                    "details": {"error": str(e)},
                    "duration_ms": 0,
                    "timestamp": time.time()
                }
                results[name] = error_result
                overall_healthy = False
                
                with self.lock:
                    self.health_status[name] = error_result
        
        return {
            "overall_healthy": overall_healthy,
            "checks": results,
            "timestamp": time.time()
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        with self.lock:
            return {
                "checks": self.health_status.copy(),
                "timestamp": time.time()
            }


class MonitoringSystem:
    """Comprehensive monitoring system."""
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self.system_monitor = SystemMonitor(self.metrics)
        self.alerts = AlertManager()
        self.health = HealthChecker()
        
        # Monitoring state
        self.running = False
        self.monitor_task: Optional[asyncio.Task] = None
        
        # Setup default health checks
        self._setup_default_health_checks()
        self._setup_default_alerts()
        
        # Add WebSocket notification callback
        self.alerts.add_callback(self._websocket_alert_callback)
    
    def _setup_default_health_checks(self):
        """Setup default health checks."""
        
        async def redis_health():
            """Check Redis connectivity."""
            try:
                async with redis_async_connection() as redis_client:
                    await redis_client.ping()
                return {"healthy": True, "message": "Redis connection OK"}
            except Exception as e:
                return {"healthy": False, "message": f"Redis connection failed: {e}"}
        
        def websocket_health():
            """Check WebSocket manager health."""
            try:
                stats = websocket_manager.get_stats()
                return {
                    "healthy": True,
                    "message": "WebSocket manager OK",
                    "details": stats
                }
            except Exception as e:
                return {"healthy": False, "message": f"WebSocket manager error: {e}"}
        
        def system_health():
            """Check system resource usage."""
            try:
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                disk_percent = psutil.disk_usage('/').percent
                
                healthy = cpu_percent < 90 and memory_percent < 90 and disk_percent < 90
                
                return {
                    "healthy": healthy,
                    "message": "System resources OK" if healthy else "High resource usage",
                    "details": {
                        "cpu_percent": cpu_percent,
                        "memory_percent": memory_percent,
                        "disk_percent": disk_percent
                    }
                }
            except Exception as e:
                return {"healthy": False, "message": f"System check failed: {e}"}
        
        self.health.register_check("redis", redis_health)
        self.health.register_check("websocket", websocket_health)
        self.health.register_check("system", system_health)
    
    def _setup_default_alerts(self):
        """Setup default alert rules."""
        
        # High CPU usage alert
        def high_cpu_condition(metrics):
            cpu_metric = metrics.get("system_cpu_percent")
            return cpu_metric and cpu_metric > 80
        
        self.alerts.add_alert_rule(
            "high_cpu_usage",
            high_cpu_condition,
            AlertSeverity.WARNING,
            "High CPU usage detected"
        )
        
        # High memory usage alert
        def high_memory_condition(metrics):
            memory_metric = metrics.get("system_memory_percent")
            return memory_metric and memory_metric > 85
        
        self.alerts.add_alert_rule(
            "high_memory_usage",
            high_memory_condition,
            AlertSeverity.WARNING,
            "High memory usage detected"
        )
        
        # Disk space alert
        def high_disk_condition(metrics):
            disk_metric = metrics.get("system_disk_percent")
            return disk_metric and disk_metric > 90
        
        self.alerts.add_alert_rule(
            "high_disk_usage",
            high_disk_condition,
            AlertSeverity.CRITICAL,
            "Critical disk space usage"
        )
    
    def _websocket_alert_callback(self, alert: Alert):
        """Send alert notifications via WebSocket."""
        try:
            asyncio.create_task(broadcast_message({
                "type": "alert",
                "data": alert.to_dict()
            }, "alerts"))
        except Exception as e:
            logger.error(f"Error broadcasting alert: {e}")
    
    async def start(self):
        """Start the monitoring system."""
        if self.running:
            return
        
        self.running = True
        
        # Start system monitoring
        await self.system_monitor.start_monitoring()
        
        # Start main monitoring loop
        self.monitor_task = asyncio.create_task(self._monitoring_loop())
        
        logger.info("Monitoring system started")
    
    async def stop(self):
        """Stop the monitoring system."""
        self.running = False
        
        # Stop system monitoring
        await self.system_monitor.stop_monitoring()
        
        # Stop main monitoring loop
        if self.monitor_task and not self.monitor_task.done():
            self.monitor_task.cancel()
            try:
                await self.monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Monitoring system stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.running:
            try:
                # Collect current metrics
                current_metrics = {}
                for metric in self.metrics.get_all_metrics():
                    current_metrics[metric["name"]] = metric["value"]
                
                # Check alerts
                self.alerts.check_alerts(current_metrics)
                
                # Run health checks periodically (every 30 seconds)
                if int(time.time()) % 30 == 0:
                    await self.health.run_checks()
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive dashboard data."""
        return {
            "metrics": self.metrics.get_all_metrics(),
            "alerts": {
                "active": [alert.to_dict() for alert in self.alerts.get_active_alerts()],
                "recent": [alert.to_dict() for alert in self.alerts.get_all_alerts()[-10:]]
            },
            "health": self.health.get_health_status(),
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "uptime": time.time() - psutil.boot_time()
            },
            "timestamp": time.time()
        }


# Global monitoring system
_monitoring_system: Optional[MonitoringSystem] = None
_monitoring_lock = threading.Lock()


def get_monitoring_system() -> MonitoringSystem:
    """Get the global monitoring system."""
    global _monitoring_system
    
    if _monitoring_system is None:
        with _monitoring_lock:
            if _monitoring_system is None:
                _monitoring_system = MonitoringSystem()
    
    return _monitoring_system


# Convenience functions for metrics
def record_counter(name: str, value: float = 1, labels: Optional[Dict[str, str]] = None):
    """Record a counter metric."""
    get_monitoring_system().metrics.record_counter(name, value, labels)


def record_gauge(name: str, value: float, labels: Optional[Dict[str, str]] = None):
    """Record a gauge metric."""
    get_monitoring_system().metrics.record_gauge(name, value, labels)


def record_histogram(name: str, value: float, labels: Optional[Dict[str, str]] = None):
    """Record a histogram metric."""
    get_monitoring_system().metrics.record_histogram(name, value, labels)


def timer(name: str, labels: Optional[Dict[str, str]] = None):
    """Context manager for timing operations."""
    class Timer:
        def __init__(self, metric_name: str, metric_labels: Optional[Dict[str, str]] = None):
            self.name = metric_name
            self.labels = metric_labels
            self.start_time = None
        
        def __enter__(self):
            self.start_time = time.time()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.start_time:
                get_monitoring_system().metrics.end_timer(self.name, self.start_time, self.labels)
    
    return Timer(name, labels)


async def start_monitoring():
    """Start the global monitoring system."""
    await get_monitoring_system().start()


async def stop_monitoring():
    """Stop the global monitoring system."""
    if _monitoring_system:
        await _monitoring_system.stop()


def get_dashboard_data() -> Dict[str, Any]:
    """Get dashboard data."""
    return get_monitoring_system().get_dashboard_data()