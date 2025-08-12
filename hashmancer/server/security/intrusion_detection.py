"""Intrusion Detection System with behavioral analysis and threat intelligence."""

import time
import json
import logging
from typing import Dict, Any, Optional, List, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import threading
import ipaddress
import re
from fastapi import Request
import hashlib

logger = logging.getLogger(__name__)


class ThreatLevel(Enum):
    """Threat severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AttackType(Enum):
    """Types of detected attacks."""
    BRUTE_FORCE = "brute_force"
    DDoS = "ddos"
    SQL_INJECTION = "sql_injection"
    XSS = "xss"
    PATH_TRAVERSAL = "path_traversal"
    COMMAND_INJECTION = "command_injection"
    BOT_ACTIVITY = "bot_activity"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"
    CREDENTIAL_STUFFING = "credential_stuffing"
    SESSION_HIJACKING = "session_hijacking"
    VULNERABILITY_SCANNING = "vulnerability_scanning"
    DATA_EXFILTRATION = "data_exfiltration"


@dataclass
class ThreatEvent:
    """Detected threat event."""
    timestamp: float
    source_ip: str
    attack_type: AttackType
    threat_level: ThreatLevel
    confidence: float
    details: Dict[str, Any]
    user_agent: Optional[str] = None
    url: Optional[str] = None
    method: Optional[str] = None
    payload: Optional[str] = None
    blocked: bool = False
    event_id: Optional[str] = None

    def __post_init__(self):
        if self.event_id is None:
            data = f"{self.timestamp}{self.source_ip}{self.attack_type.value}"
            self.event_id = hashlib.sha256(data.encode()).hexdigest()[:16]


@dataclass
class BehaviorProfile:
    """User/IP behavior profile for anomaly detection."""
    ip_address: str
    first_seen: float
    last_seen: float
    request_count: int
    unique_urls: Set[str]
    unique_user_agents: Set[str]
    methods_used: Set[str]
    avg_request_rate: float
    typical_hours: Set[int]  # Hours of day when typically active
    countries_seen: Set[str]
    failed_logins: int
    successful_logins: int


class IntrusionDetectionSystem:
    """Advanced intrusion detection with behavioral analysis."""

    def __init__(self):
        self._threat_events: deque = deque(maxlen=10000)
        self._behavior_profiles: Dict[str, BehaviorProfile] = {}
        self._known_bad_ips: Set[str] = set()
        self._honeypots: Set[str] = set()
        self._lock = threading.Lock()
        
        # Detection patterns
        self._attack_patterns = {
            AttackType.SQL_INJECTION: [
                re.compile(r'(?i)(union\s+select|drop\s+table|insert\s+into|delete\s+from)', re.I),
                re.compile(r'(?i)(or\s+1\s*=\s*1|and\s+1\s*=\s*1)', re.I),
                re.compile(r'(?i)(\'|\").*(\;|--|\#)', re.I),
                re.compile(r'(?i)(exec\s*\(|sp_|xp_)', re.I),
            ],
            AttackType.XSS: [
                re.compile(r'(?i)(<script|javascript:|vbscript:|onload\s*=|onerror\s*=)', re.I),
                re.compile(r'(?i)(<iframe|<object|<embed|<img[^>]*src\s*=)', re.I),
                re.compile(r'(?i)(alert\s*\(|confirm\s*\(|prompt\s*\()', re.I),
            ],
            AttackType.PATH_TRAVERSAL: [
                re.compile(r'(?i)(\.\.\/|\.\.\\|\/etc\/passwd|\/proc\/|\/sys\/)', re.I),
                re.compile(r'(?i)(\.\.%2f|\.\.%5c|%2e%2e%2f)', re.I),
            ],
            AttackType.COMMAND_INJECTION: [
                re.compile(r'(?i)(\||\;|&&|\$\(|`)', re.I),
                re.compile(r'(?i)(cmd\s*=|exec\s*=|system\s*\(|passthru\s*\()', re.I),
                re.compile(r'(?i)(nc\s|netcat|wget\s|curl\s)', re.I),
            ],
        }
        
        # Vulnerability scanning patterns
        self._vuln_scan_patterns = [
            re.compile(r'(?i)(nikto|nessus|openvas|nmap|sqlmap)', re.I),
            re.compile(r'(?i)(\.(php|asp|jsp|cgi)\~|\.(bak|old|orig))', re.I),
            re.compile(r'(?i)(\/admin|\/config|\/backup|\/test)', re.I),
        ]
        
        # Bot detection patterns
        self._bot_patterns = [
            re.compile(r'(?i)(bot|crawler|spider|scraper)', re.I),
            re.compile(r'(?i)(python-requests|curl\/|wget\/)', re.I),
        ]
        
        # Honeypot URLs (fake sensitive endpoints)
        self._honeypots = {
            '/admin.php', '/admin/', '/.env', '/config.php', 
            '/backup/', '/test.php', '/phpinfo.php', '/wp-admin/',
            '/.git/config', '/server-status', '/server-info'
        }
        
        # Threat intelligence (simplified - in production would integrate with feeds)
        self._threat_intel = {
            'known_bad_ips': set(),
            'tor_exit_nodes': set(),
            'malicious_user_agents': {
                'masscan', 'zmap', 'sqlmap', 'nikto', 'dirb', 'gobuster'
            }
        }
        
    def analyze_request(self, request: Request, response_code: Optional[int] = None) -> Optional[ThreatEvent]:
        """Analyze incoming request for threats."""
        current_time = time.time()
        source_ip = self._extract_ip(request)
        url = str(request.url)
        method = request.method
        user_agent = request.headers.get('User-Agent', '')
        
        # Update behavior profile
        self._update_behavior_profile(source_ip, request, current_time)
        
        # Check various threat types
        threat_event = None
        
        # 1. Check honeypots
        if any(honeypot in url for honeypot in self._honeypots):
            threat_event = ThreatEvent(
                timestamp=current_time,
                source_ip=source_ip,
                attack_type=AttackType.VULNERABILITY_SCANNING,
                threat_level=ThreatLevel.HIGH,
                confidence=0.9,
                details={'honeypot_accessed': url},
                user_agent=user_agent,
                url=url,
                method=method
            )
        
        # 2. Check attack patterns
        if not threat_event:
            for attack_type, patterns in self._attack_patterns.items():
                for pattern in patterns:
                    if pattern.search(url) or pattern.search(user_agent):
                        threat_event = ThreatEvent(
                            timestamp=current_time,
                            source_ip=source_ip,
                            attack_type=attack_type,
                            threat_level=ThreatLevel.HIGH,
                            confidence=0.8,
                            details={'pattern_matched': pattern.pattern},
                            user_agent=user_agent,
                            url=url,
                            method=method,
                            payload=url
                        )
                        break
                if threat_event:
                    break
        
        # 3. Check for vulnerability scanning
        if not threat_event:
            for pattern in self._vuln_scan_patterns:
                if pattern.search(url) or pattern.search(user_agent):
                    threat_event = ThreatEvent(
                        timestamp=current_time,
                        source_ip=source_ip,
                        attack_type=AttackType.VULNERABILITY_SCANNING,
                        threat_level=ThreatLevel.MEDIUM,
                        confidence=0.7,
                        details={'scan_pattern': pattern.pattern},
                        user_agent=user_agent,
                        url=url,
                        method=method
                    )
                    break
        
        # 4. Check for bot activity
        if not threat_event:
            for pattern in self._bot_patterns:
                if pattern.search(user_agent):
                    # Not all bots are threats, but worth monitoring
                    if not any(good_bot in user_agent.lower() for good_bot in 
                             ['googlebot', 'bingbot', 'slackbot', 'twitterbot']):
                        threat_event = ThreatEvent(
                            timestamp=current_time,
                            source_ip=source_ip,
                            attack_type=AttackType.BOT_ACTIVITY,
                            threat_level=ThreatLevel.LOW,
                            confidence=0.6,
                            details={'bot_pattern': pattern.pattern},
                            user_agent=user_agent,
                            url=url,
                            method=method
                        )
                    break
        
        # 5. Behavioral anomaly detection
        if not threat_event:
            anomaly = self._detect_behavioral_anomaly(source_ip, request, current_time)
            if anomaly:
                threat_event = anomaly
        
        # 6. Check threat intelligence
        if not threat_event:
            intel_threat = self._check_threat_intelligence(source_ip, user_agent)
            if intel_threat:
                threat_event = intel_threat
                threat_event.timestamp = current_time
                threat_event.url = url
                threat_event.method = method
        
        # Record threat event if detected
        if threat_event:
            with self._lock:
                self._threat_events.append(threat_event)
            
            logger.warning(f"Threat detected: {threat_event.attack_type.value} from {source_ip}")
            return threat_event
        
        return None
    
    def _extract_ip(self, request: Request) -> str:
        """Extract real IP address from request."""
        forwarded_for = request.headers.get('X-Forwarded-For')
        if forwarded_for:
            return forwarded_for.split(',')[0].strip()
        
        real_ip = request.headers.get('X-Real-IP')
        if real_ip:
            return real_ip.strip()
        
        if request.client:
            return request.client.host
        
        return "unknown"
    
    def _update_behavior_profile(self, ip: str, request: Request, timestamp: float):
        """Update behavioral profile for IP."""
        with self._lock:
            if ip not in self._behavior_profiles:
                self._behavior_profiles[ip] = BehaviorProfile(
                    ip_address=ip,
                    first_seen=timestamp,
                    last_seen=timestamp,
                    request_count=0,
                    unique_urls=set(),
                    unique_user_agents=set(),
                    methods_used=set(),
                    avg_request_rate=0.0,
                    typical_hours=set(),
                    countries_seen=set(),
                    failed_logins=0,
                    successful_logins=0
                )
            
            profile = self._behavior_profiles[ip]
            profile.last_seen = timestamp
            profile.request_count += 1
            profile.unique_urls.add(str(request.url.path))
            profile.unique_user_agents.add(request.headers.get('User-Agent', 'unknown'))
            profile.methods_used.add(request.method)
            profile.typical_hours.add(time.localtime(timestamp).tm_hour)
            
            # Update request rate
            time_active = max(1, timestamp - profile.first_seen)
            profile.avg_request_rate = profile.request_count / time_active
    
    def _detect_behavioral_anomaly(self, ip: str, request: Request, timestamp: float) -> Optional[ThreatEvent]:
        """Detect behavioral anomalies."""
        profile = self._behavior_profiles.get(ip)
        if not profile or profile.request_count < 10:  # Need baseline
            return None
        
        anomalies = []
        
        # Check request rate anomaly
        recent_window = 300  # 5 minutes
        if profile.avg_request_rate > 0:
            current_rate = profile.request_count / max(1, timestamp - profile.first_seen)
            if current_rate > profile.avg_request_rate * 10:  # 10x normal rate
                anomalies.append("excessive_request_rate")
        
        # Check for unusual URL patterns
        current_path = str(request.url.path)
        if len(profile.unique_urls) < 5 and current_path not in profile.unique_urls:
            # New IP accessing many different URLs quickly
            anomalies.append("url_scanning")
        
        # Check for time-based anomalies
        current_hour = time.localtime(timestamp).tm_hour
        if (len(profile.typical_hours) > 5 and 
            current_hour not in profile.typical_hours and 
            profile.request_count > 100):
            anomalies.append("unusual_time_activity")
        
        if anomalies:
            return ThreatEvent(
                timestamp=timestamp,
                source_ip=ip,
                attack_type=AttackType.ANOMALOUS_BEHAVIOR,
                threat_level=ThreatLevel.MEDIUM,
                confidence=0.6,
                details={'anomalies': anomalies},
                user_agent=request.headers.get('User-Agent'),
                url=str(request.url),
                method=request.method
            )
        
        return None
    
    def _check_threat_intelligence(self, ip: str, user_agent: str) -> Optional[ThreatEvent]:
        """Check against threat intelligence feeds."""
        # Check known bad IPs
        if ip in self._threat_intel['known_bad_ips']:
            return ThreatEvent(
                timestamp=time.time(),
                source_ip=ip,
                attack_type=AttackType.ANOMALOUS_BEHAVIOR,
                threat_level=ThreatLevel.CRITICAL,
                confidence=0.95,
                details={'threat_intel': 'known_bad_ip'},
                user_agent=user_agent
            )
        
        # Check malicious user agents
        ua_lower = user_agent.lower()
        for bad_ua in self._threat_intel['malicious_user_agents']:
            if bad_ua in ua_lower:
                return ThreatEvent(
                    timestamp=time.time(),
                    source_ip=ip,
                    attack_type=AttackType.VULNERABILITY_SCANNING,
                    threat_level=ThreatLevel.HIGH,
                    confidence=0.9,
                    details={'threat_intel': 'malicious_user_agent', 'pattern': bad_ua},
                    user_agent=user_agent
                )
        
        return None
    
    def get_threat_events(
        self, 
        limit: int = 100,
        threat_level: Optional[ThreatLevel] = None,
        attack_type: Optional[AttackType] = None,
        source_ip: Optional[str] = None,
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """Get recent threat events with filtering."""
        cutoff = time.time() - (hours * 3600)
        
        with self._lock:
            events = [e for e in self._threat_events if e.timestamp >= cutoff]
        
        if threat_level:
            events = [e for e in events if e.threat_level == threat_level]
        
        if attack_type:
            events = [e for e in events if e.attack_type == attack_type]
        
        if source_ip:
            events = [e for e in events if e.source_ip == source_ip]
        
        events.sort(key=lambda e: e.timestamp, reverse=True)
        return [asdict(event) for event in events[:limit]]
    
    def get_threat_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get threat detection statistics."""
        cutoff = time.time() - (hours * 3600)
        
        with self._lock:
            recent_events = [e for e in self._threat_events if e.timestamp >= cutoff]
        
        # Attack type distribution
        attack_counts = defaultdict(int)
        threat_level_counts = defaultdict(int)
        top_attackers = defaultdict(int)
        
        for event in recent_events:
            attack_counts[event.attack_type.value] += 1
            threat_level_counts[event.threat_level.value] += 1
            top_attackers[event.source_ip] += 1
        
        # Get top attackers
        top_attackers_list = sorted(top_attackers.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # Blocked vs detected
        blocked_count = sum(1 for e in recent_events if e.blocked)
        
        return {
            'time_period_hours': hours,
            'total_threats': len(recent_events),
            'threats_blocked': blocked_count,
            'threats_detected': len(recent_events) - blocked_count,
            'attack_type_distribution': dict(attack_counts),
            'threat_level_distribution': dict(threat_level_counts),
            'top_attacking_ips': top_attackers_list,
            'active_behavior_profiles': len(self._behavior_profiles),
            'avg_threats_per_hour': len(recent_events) / max(hours, 1)
        }
    
    def add_to_threat_intel(self, ip: str, reason: str = "Manual addition"):
        """Add IP to threat intelligence blacklist."""
        with self._lock:
            self._threat_intel['known_bad_ips'].add(ip)
        logger.info(f"Added {ip} to threat intelligence blacklist: {reason}")
    
    def remove_from_threat_intel(self, ip: str):
        """Remove IP from threat intelligence blacklist."""
        with self._lock:
            self._threat_intel['known_bad_ips'].discard(ip)
        logger.info(f"Removed {ip} from threat intelligence blacklist")
    
    def get_behavior_profile(self, ip: str) -> Optional[Dict[str, Any]]:
        """Get behavior profile for IP."""
        profile = self._behavior_profiles.get(ip)
        if not profile:
            return None
        
        return {
            'ip_address': profile.ip_address,
            'first_seen': profile.first_seen,
            'last_seen': profile.last_seen,
            'request_count': profile.request_count,
            'unique_urls_count': len(profile.unique_urls),
            'unique_user_agents_count': len(profile.unique_user_agents),
            'methods_used': list(profile.methods_used),
            'avg_request_rate': profile.avg_request_rate,
            'typical_hours': list(profile.typical_hours),
            'failed_logins': profile.failed_logins,
            'successful_logins': profile.successful_logins
        }
    
    def cleanup_old_data(self, max_age: int = 86400 * 7):  # 7 days
        """Clean up old threat data and behavior profiles."""
        cutoff = time.time() - max_age
        
        with self._lock:
            # Remove old behavior profiles
            old_profiles = [ip for ip, profile in self._behavior_profiles.items() 
                           if profile.last_seen < cutoff]
            
            for ip in old_profiles:
                del self._behavior_profiles[ip]
            
            logger.debug(f"Cleaned up {len(old_profiles)} old behavior profiles")


# Global IDS instance
_ids: Optional[IntrusionDetectionSystem] = None


def get_intrusion_detection_system() -> IntrusionDetectionSystem:
    """Get global intrusion detection system instance."""
    global _ids
    if _ids is None:
        _ids = IntrusionDetectionSystem()
    return _ids


def analyze_request_for_threats(request: Request, response_code: Optional[int] = None) -> Optional[ThreatEvent]:
    """Analyze request for threats using global IDS."""
    return get_intrusion_detection_system().analyze_request(request, response_code)


def get_threat_events(**kwargs) -> List[Dict[str, Any]]:
    """Get threat events with filtering."""
    return get_intrusion_detection_system().get_threat_events(**kwargs)


def get_threat_statistics(hours: int = 24) -> Dict[str, Any]:
    """Get threat detection statistics."""
    return get_intrusion_detection_system().get_threat_statistics(hours)