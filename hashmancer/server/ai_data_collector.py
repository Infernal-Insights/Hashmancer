"""Data collection pipeline for AI model training in Hashmancer.

This module collects and preprocesses data from active hash cracking operations
to train and improve AI models for pattern recognition and strategy optimization.
"""

import json
import time
import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import random

from .redis_utils import get_redis
from .pattern_stats import TOKEN_RE
from .pattern_utils import word_to_pattern, is_valid_word
from hashmancer.utils.event_logger import log_info


@dataclass
class CrackingEvent:
    """Data structure for a single password cracking event."""
    timestamp: float
    hash_value: str
    password: str
    hash_type: str
    strategy_used: str
    time_to_crack: float
    gpu_utilization: float
    success: bool
    pattern: str
    attempts_before_success: int = 0
    worker_id: str = ""
    batch_size: int = 0


@dataclass
class StrategyPerformance:
    """Performance metrics for a specific strategy."""
    strategy_name: str
    total_attempts: int
    successful_cracks: int
    average_time: float
    success_rate: float
    gpu_efficiency: float
    hash_types: List[str]
    timestamp: float


@dataclass
class PatternTransition:
    """Password pattern transition for training."""
    from_pattern: str
    to_pattern: str
    frequency: int
    context: Dict[str, Any]
    timestamp: float


class AIDataCollector:
    """Collects and manages training data for AI models."""
    
    def __init__(self, data_dir: str = "ai_training_data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.redis_client = get_redis()
        self.collection_enabled = True
        self.privacy_mode = True  # Anonymize sensitive data
        
        # Data collection buffers
        self.cracking_events: List[CrackingEvent] = []
        self.strategy_metrics: Dict[str, StrategyPerformance] = {}
        self.pattern_transitions: List[PatternTransition] = []
        
        # Collection limits
        self.max_buffer_size = 1000
        self.flush_interval = 300  # 5 minutes
        
        # File paths
        self.cracking_events_file = self.data_dir / "cracking_events.jsonl"
        self.strategy_metrics_file = self.data_dir / "strategy_metrics.jsonl" 
        self.pattern_transitions_file = self.data_dir / "pattern_transitions.jsonl"
        
        logging.info(f"AI Data Collector initialized with data dir: {self.data_dir}")
    
    async def collect_cracking_event(
        self,
        hash_value: str,
        password: str,
        hash_type: str,
        strategy_used: str,
        time_to_crack: float,
        success: bool,
        **kwargs
    ):
        """Collect data from a password cracking event."""
        if not self.collection_enabled:
            return
        
        try:
            # Generate pattern from password
            pattern = word_to_pattern(password) if password else ""
            
            # Anonymize sensitive data if privacy mode is enabled
            if self.privacy_mode:
                hash_value = self._anonymize_hash(hash_value)
                password = self._anonymize_password(password)
            
            event = CrackingEvent(
                timestamp=time.time(),
                hash_value=hash_value,
                password=password,
                hash_type=hash_type,
                strategy_used=strategy_used,
                time_to_crack=time_to_crack,
                gpu_utilization=kwargs.get('gpu_utilization', 0.0),
                success=success,
                pattern=pattern,
                attempts_before_success=kwargs.get('attempts_before_success', 0),
                worker_id=kwargs.get('worker_id', ''),
                batch_size=kwargs.get('batch_size', 0)
            )
            
            self.cracking_events.append(event)
            
            # Update strategy performance metrics
            await self._update_strategy_metrics(event)
            
            # Collect pattern transitions if successful
            if success and pattern:
                await self._collect_pattern_transitions(pattern)
            
            # Store in Redis for real-time access
            await self._store_event_in_redis(event)
            
            # Flush buffers if they're getting large
            if len(self.cracking_events) >= self.max_buffer_size:
                await self.flush_data()
                
        except Exception as e:
            logging.error(f"Error collecting cracking event: {e}")
    
    async def collect_strategy_switch(
        self,
        old_strategy: str,
        new_strategy: str,
        reason: str,
        performance_delta: float
    ):
        """Collect data when AI engine switches strategies."""
        try:
            switch_event = {
                "timestamp": time.time(),
                "old_strategy": old_strategy,
                "new_strategy": new_strategy,
                "reason": reason,
                "performance_delta": performance_delta,
                "event_type": "strategy_switch"
            }
            
            # Store in Redis
            self.redis_client.lpush("ai:strategy_switches", json.dumps(switch_event))
            self.redis_client.ltrim("ai:strategy_switches", 0, 999)  # Keep last 1000
            
        except Exception as e:
            logging.error(f"Error collecting strategy switch: {e}")
    
    async def collect_pattern_prediction_feedback(
        self,
        predicted_patterns: List[Tuple[str, float]],
        actual_pattern: str,
        prediction_accuracy: float
    ):
        """Collect feedback on pattern prediction accuracy."""
        try:
            feedback = {
                "timestamp": time.time(),
                "predicted_patterns": predicted_patterns,
                "actual_pattern": actual_pattern,
                "prediction_accuracy": prediction_accuracy,
                "event_type": "prediction_feedback"
            }
            
            # Store in Redis
            self.redis_client.lpush("ai:prediction_feedback", json.dumps(feedback))
            self.redis_client.ltrim("ai:prediction_feedback", 0, 999)
            
        except Exception as e:
            logging.error(f"Error collecting prediction feedback: {e}")
    
    async def _update_strategy_metrics(self, event: CrackingEvent):
        """Update strategy performance metrics."""
        strategy = event.strategy_used
        
        if strategy not in self.strategy_metrics:
            self.strategy_metrics[strategy] = StrategyPerformance(
                strategy_name=strategy,
                total_attempts=0,
                successful_cracks=0,
                average_time=0.0,
                success_rate=0.0,
                gpu_efficiency=0.0,
                hash_types=[],
                timestamp=time.time()
            )
        
        metrics = self.strategy_metrics[strategy]
        metrics.total_attempts += 1
        
        if event.success:
            metrics.successful_cracks += 1
            # Update average time (running average)
            metrics.average_time = (
                (metrics.average_time * (metrics.successful_cracks - 1) + event.time_to_crack) /
                metrics.successful_cracks
            )
        
        metrics.success_rate = metrics.successful_cracks / metrics.total_attempts
        
        # Update GPU efficiency (running average)
        metrics.gpu_efficiency = (
            (metrics.gpu_efficiency * (metrics.total_attempts - 1) + event.gpu_utilization) /
            metrics.total_attempts
        )
        
        # Add hash type if not already tracked
        if event.hash_type not in metrics.hash_types:
            metrics.hash_types.append(event.hash_type)
        
        metrics.timestamp = time.time()
    
    async def _collect_pattern_transitions(self, current_pattern: str):
        """Collect pattern transitions for sequence learning."""
        try:
            # Get recent patterns from Redis
            recent_patterns = self.redis_client.lrange("ai:recent_patterns", 0, 10)
            
            if recent_patterns:
                previous_pattern = recent_patterns[0]
                if isinstance(previous_pattern, bytes):
                    previous_pattern = previous_pattern.decode()
                
                if previous_pattern and previous_pattern != current_pattern:
                    transition = PatternTransition(
                        from_pattern=previous_pattern,
                        to_pattern=current_pattern,
                        frequency=1,
                        context={"timestamp": time.time()},
                        timestamp=time.time()
                    )
                    
                    self.pattern_transitions.append(transition)
            
            # Update recent patterns list
            self.redis_client.lpush("ai:recent_patterns", current_pattern)
            self.redis_client.ltrim("ai:recent_patterns", 0, 99)  # Keep last 100
            
        except Exception as e:
            logging.error(f"Error collecting pattern transitions: {e}")
    
    async def _store_event_in_redis(self, event: CrackingEvent):
        """Store event in Redis for real-time access."""
        try:
            event_data = json.dumps(asdict(event))
            
            # Store in multiple Redis structures for different access patterns
            self.redis_client.lpush("ai:recent_events", event_data)
            self.redis_client.ltrim("ai:recent_events", 0, 999)  # Keep last 1000
            
            # Store by strategy for strategy-specific analysis
            self.redis_client.lpush(f"ai:events:{event.strategy_used}", event_data)
            self.redis_client.ltrim(f"ai:events:{event.strategy_used}", 0, 499)
            
            # Store by hash type
            self.redis_client.lpush(f"ai:events:hash:{event.hash_type}", event_data)
            self.redis_client.ltrim(f"ai:events:hash:{event.hash_type}", 0, 499)
            
        except Exception as e:
            logging.error(f"Error storing event in Redis: {e}")
    
    def _anonymize_hash(self, hash_value: str) -> str:
        """Anonymize hash value for privacy."""
        if not hash_value:
            return ""
        
        # Create a stable but anonymized version
        salt = "hashmancer_ai_training"
        return hashlib.sha256(f"{salt}:{hash_value}".encode()).hexdigest()[:16]
    
    def _anonymize_password(self, password: str) -> str:
        """Anonymize password while preserving pattern."""
        if not password:
            return ""
        
        # Replace actual characters with pattern placeholders
        # This preserves the structure for AI training while removing sensitive content
        pattern = word_to_pattern(password)
        
        # Generate a pseudo-password from the pattern for training
        anonymized = ""
        for char in password:
            if char.islower():
                anonymized += chr(ord('a') + (hash(char) % 26))
            elif char.isupper():
                anonymized += chr(ord('A') + (hash(char) % 26))
            elif char.isdigit():
                anonymized += str(hash(char) % 10)
            else:
                anonymized += char  # Keep special characters
        
        return anonymized
    
    async def flush_data(self):
        """Flush collected data to persistent storage."""
        try:
            # Write cracking events
            if self.cracking_events:
                await self._write_jsonl_data(self.cracking_events_file, self.cracking_events)
                logging.info(f"AI Data Collector: Flushed {len(self.cracking_events)} cracking events")
                self.cracking_events.clear()
            
            # Write strategy metrics
            if self.strategy_metrics:
                metrics_list = list(self.strategy_metrics.values())
                await self._write_jsonl_data(self.strategy_metrics_file, metrics_list)
                logging.info(f"AI Data Collector: Flushed {len(metrics_list)} strategy metrics")
            
            # Write pattern transitions
            if self.pattern_transitions:
                await self._write_jsonl_data(self.pattern_transitions_file, self.pattern_transitions)
                logging.info(f"AI Data Collector: Flushed {len(self.pattern_transitions)} pattern transitions")
                self.pattern_transitions.clear()
                
        except Exception as e:
            logging.error(f"Error flushing data: {e}")
    
    async def _write_jsonl_data(self, file_path: Path, data: List[Any]):
        """Write data to JSONL file."""
        try:
            with open(file_path, 'a', encoding='utf-8') as f:
                for item in data:
                    if hasattr(item, '__dict__'):
                        json_line = json.dumps(asdict(item))
                    else:
                        json_line = json.dumps(item)
                    f.write(json_line + '\n')
        except Exception as e:
            logging.error(f"Error writing to {file_path}: {e}")
    
    async def get_training_dataset(
        self,
        event_type: str = "cracking_events",
        limit: int = 10000,
        hash_type_filter: Optional[str] = None,
        strategy_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get training dataset for AI models."""
        try:
            if event_type == "cracking_events":
                file_path = self.cracking_events_file
            elif event_type == "strategy_metrics":
                file_path = self.strategy_metrics_file
            elif event_type == "pattern_transitions":
                file_path = self.pattern_transitions_file
            else:
                raise ValueError(f"Unknown event type: {event_type}")
            
            if not file_path.exists():
                return []
            
            dataset = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    if line_num >= limit:
                        break
                    
                    try:
                        data = json.loads(line.strip())
                        
                        # Apply filters
                        if hash_type_filter and data.get('hash_type') != hash_type_filter:
                            continue
                        if strategy_filter and data.get('strategy_used') != strategy_filter:
                            continue
                        
                        dataset.append(data)
                    except json.JSONDecodeError:
                        continue
            
            return dataset
            
        except Exception as e:
            logging.error(f"Error getting training dataset: {e}")
            return []
    
    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about collected data."""
        try:
            stats = {
                "collection_enabled": self.collection_enabled,
                "privacy_mode": self.privacy_mode,
                "buffer_sizes": {
                    "cracking_events": len(self.cracking_events),
                    "pattern_transitions": len(self.pattern_transitions),
                    "strategy_metrics": len(self.strategy_metrics)
                },
                "file_sizes": {},
                "redis_stats": {}
            }
            
            # Get file sizes
            for file_name, file_path in [
                ("cracking_events", self.cracking_events_file),
                ("strategy_metrics", self.strategy_metrics_file),
                ("pattern_transitions", self.pattern_transitions_file)
            ]:
                if file_path.exists():
                    stats["file_sizes"][file_name] = file_path.stat().st_size
                else:
                    stats["file_sizes"][file_name] = 0
            
            # Get Redis stats
            redis_keys = [
                "ai:recent_events",
                "ai:recent_patterns", 
                "ai:strategy_switches",
                "ai:prediction_feedback"
            ]
            
            for key in redis_keys:
                try:
                    stats["redis_stats"][key] = self.redis_client.llen(key)
                except:
                    stats["redis_stats"][key] = 0
            
            return stats
            
        except Exception as e:
            logging.error(f"Error getting collection stats: {e}")
            return {"error": str(e)}
    
    async def start_background_collection(self):
        """Start background data collection and periodic flushing."""
        logging.info("Starting AI data collection background task")
        
        while self.collection_enabled:
            try:
                await asyncio.sleep(self.flush_interval)
                await self.flush_data()
            except Exception as e:
                logging.error(f"Background collection error: {e}")
                await asyncio.sleep(60)  # Wait before retrying


# Global data collector instance
_data_collector: Optional[AIDataCollector] = None


def get_ai_data_collector() -> AIDataCollector:
    """Get or create the global AI data collector instance."""
    global _data_collector
    if _data_collector is None:
        _data_collector = AIDataCollector()
    return _data_collector


async def collect_cracking_success(
    hash_value: str,
    password: str,
    hash_type: str,
    strategy_used: str,
    time_to_crack: float,
    **kwargs
):
    """Convenience function to collect successful cracking events."""
    collector = get_ai_data_collector()
    await collector.collect_cracking_event(
        hash_value=hash_value,
        password=password,
        hash_type=hash_type,
        strategy_used=strategy_used,
        time_to_crack=time_to_crack,
        success=True,
        **kwargs
    )


async def collect_cracking_failure(
    hash_value: str,
    hash_type: str,
    strategy_used: str,
    time_attempted: float,
    **kwargs
):
    """Convenience function to collect failed cracking attempts."""
    collector = get_ai_data_collector()
    await collector.collect_cracking_event(
        hash_value=hash_value,
        password="",  # No password for failed attempts
        hash_type=hash_type,
        strategy_used=strategy_used,
        time_to_crack=time_attempted,
        success=False,
        **kwargs
    )