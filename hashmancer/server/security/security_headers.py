"""Security headers middleware for web application hardening."""

import time
from typing import Dict, Any, Optional, List
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response as StarletteResponse
import secrets


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add comprehensive security headers."""
    
    def __init__(self, app, config: Optional[Dict[str, Any]] = None):
        super().__init__(app)
        self.config = config or {}
        
        # Default security configuration
        self.default_config = {
            'hsts_enabled': True,
            'hsts_max_age': 31536000,  # 1 year
            'hsts_include_subdomains': True,
            'hsts_preload': False,
            
            'csp_enabled': True,
            'csp_default_src': "'self'",
            'csp_script_src': "'self' 'unsafe-inline' https://fonts.googleapis.com",
            'csp_style_src': "'self' 'unsafe-inline' https://fonts.googleapis.com",
            'csp_img_src': "'self' data: https:",
            'csp_connect_src': "'self' ws: wss:",
            'csp_font_src': "'self' https://fonts.gstatic.com",
            'csp_report_uri': None,
            
            'x_frame_options': 'DENY',
            'x_content_type_options': 'nosniff',
            'x_xss_protection': '1; mode=block',
            'referrer_policy': 'strict-origin-when-cross-origin',
            
            'permissions_policy_enabled': True,
            'permissions_policy': {
                'camera': '()',
                'microphone': '()',
                'geolocation': '()',
                'payment': '()',
                'usb': '()',
                'magnetometer': '()',
                'gyroscope': '()',
                'accelerometer': '()',
            },
            
            'remove_server_header': True,
            'remove_x_powered_by': True,
            
            'expect_ct_enabled': False,
            'expect_ct_max_age': 86400,
            'expect_ct_enforce': False,
            'expect_ct_report_uri': None,
        }
        
        # Merge with provided config
        self.security_config = {**self.default_config, **self.config}
        
        # Generate nonce for CSP if needed
        self._nonce = None
    
    def generate_csp_nonce(self) -> str:
        """Generate a new CSP nonce."""
        return secrets.token_urlsafe(16)
    
    async def dispatch(self, request: Request, call_next):
        """Add security headers to response."""
        # Generate nonce for this request
        self._nonce = self.generate_csp_nonce()
        
        # Store nonce in request state for use in templates
        request.state.csp_nonce = self._nonce
        
        response = await call_next(request)
        
        # Add security headers
        self.add_security_headers(response, request)
        
        return response
    
    def add_security_headers(self, response: Response, request: Request):
        """Add all configured security headers."""
        
        # HTTP Strict Transport Security (HSTS)
        if self.security_config['hsts_enabled'] and request.url.scheme == 'https':
            hsts_value = f"max-age={self.security_config['hsts_max_age']}"
            if self.security_config['hsts_include_subdomains']:
                hsts_value += "; includeSubDomains"
            if self.security_config['hsts_preload']:
                hsts_value += "; preload"
            response.headers['Strict-Transport-Security'] = hsts_value
        
        # Content Security Policy (CSP)
        if self.security_config['csp_enabled']:
            csp_directives = []
            
            # Default source
            csp_directives.append(f"default-src {self.security_config['csp_default_src']}")
            
            # Script source with nonce
            script_src = self.security_config['csp_script_src']
            if "'unsafe-inline'" not in script_src:
                script_src += f" 'nonce-{self._nonce}'"
            csp_directives.append(f"script-src {script_src}")
            
            # Style source
            csp_directives.append(f"style-src {self.security_config['csp_style_src']}")
            
            # Image source
            csp_directives.append(f"img-src {self.security_config['csp_img_src']}")
            
            # Connect source (for AJAX/WebSocket)
            csp_directives.append(f"connect-src {self.security_config['csp_connect_src']}")
            
            # Font source
            csp_directives.append(f"font-src {self.security_config['csp_font_src']}")
            
            # Object and embed sources (block plugins)
            csp_directives.append("object-src 'none'")
            csp_directives.append("embed-src 'none'")
            
            # Base URI
            csp_directives.append("base-uri 'self'")
            
            # Form action
            csp_directives.append("form-action 'self'")
            
            # Frame ancestors (clickjacking protection)
            csp_directives.append("frame-ancestors 'none'")
            
            # Report URI
            if self.security_config['csp_report_uri']:
                csp_directives.append(f"report-uri {self.security_config['csp_report_uri']}")
            
            csp_header = "; ".join(csp_directives)
            response.headers['Content-Security-Policy'] = csp_header
        
        # X-Frame-Options (clickjacking protection)
        if self.security_config['x_frame_options']:
            response.headers['X-Frame-Options'] = self.security_config['x_frame_options']
        
        # X-Content-Type-Options (MIME sniffing protection)
        if self.security_config['x_content_type_options']:
            response.headers['X-Content-Type-Options'] = self.security_config['x_content_type_options']
        
        # X-XSS-Protection (XSS filtering)
        if self.security_config['x_xss_protection']:
            response.headers['X-XSS-Protection'] = self.security_config['x_xss_protection']
        
        # Referrer Policy
        if self.security_config['referrer_policy']:
            response.headers['Referrer-Policy'] = self.security_config['referrer_policy']
        
        # Permissions Policy (formerly Feature Policy)
        if self.security_config['permissions_policy_enabled']:
            permissions = []
            for feature, allowlist in self.security_config['permissions_policy'].items():
                permissions.append(f"{feature}={allowlist}")
            if permissions:
                response.headers['Permissions-Policy'] = ", ".join(permissions)
        
        # Expect-CT (Certificate Transparency)
        if self.security_config['expect_ct_enabled']:
            expect_ct_value = f"max-age={self.security_config['expect_ct_max_age']}"
            if self.security_config['expect_ct_enforce']:
                expect_ct_value += ", enforce"
            if self.security_config['expect_ct_report_uri']:
                expect_ct_value += f", report-uri=\"{self.security_config['expect_ct_report_uri']}\""
            response.headers['Expect-CT'] = expect_ct_value
        
        # Remove identifying server headers
        if self.security_config['remove_server_header']:
            response.headers.pop('Server', None)
        
        if self.security_config['remove_x_powered_by']:
            response.headers.pop('X-Powered-By', None)
        
        # Additional security headers
        response.headers['X-DNS-Prefetch-Control'] = 'off'
        response.headers['X-Download-Options'] = 'noopen'
        response.headers['X-Permitted-Cross-Domain-Policies'] = 'none'
        
        # Cache control for sensitive endpoints
        if self._is_sensitive_endpoint(request.url.path):
            response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, private'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
    
    def _is_sensitive_endpoint(self, path: str) -> bool:
        """Check if endpoint contains sensitive data."""
        sensitive_patterns = [
            '/login',
            '/logout',
            '/admin',
            '/api/',
            '/auth',
            '/password',
            '/register',
            '/settings'
        ]
        return any(pattern in path for pattern in sensitive_patterns)


def add_security_headers(response: StarletteResponse, request: Request) -> StarletteResponse:
    """Standalone function to add security headers to a response."""
    middleware = SecurityHeadersMiddleware(None)
    middleware.add_security_headers(response, request)
    return response


def get_csp_nonce(request: Request) -> Optional[str]:
    """Get CSP nonce from request state."""
    return getattr(request.state, 'csp_nonce', None)


def create_security_config(
    environment: str = 'production',
    custom_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create security configuration based on environment."""
    
    if environment == 'development':
        config = {
            'hsts_enabled': False,
            'csp_enabled': False,  # Disable CSP in dev for easier debugging
            'x_frame_options': 'SAMEORIGIN',  # Allow framing in dev
            'remove_server_header': False,
            'expect_ct_enabled': False,
        }
    elif environment == 'staging':
        config = {
            'hsts_enabled': True,
            'hsts_max_age': 300,  # 5 minutes for testing
            'csp_enabled': True,
            'csp_report_uri': '/csp-report',  # Enable reporting in staging
            'x_frame_options': 'DENY',
        }
    else:  # production
        config = {
            'hsts_enabled': True,
            'hsts_max_age': 31536000,  # 1 year
            'hsts_include_subdomains': True,
            'hsts_preload': True,
            'csp_enabled': True,
            'csp_report_uri': '/csp-report',
            'expect_ct_enabled': True,
            'expect_ct_enforce': True,
        }
    
    if custom_config:
        config.update(custom_config)
    
    return config


def validate_security_headers(response_headers: Dict[str, str]) -> Dict[str, Any]:
    """Validate that security headers are properly set."""
    results = {
        'score': 0,
        'max_score': 100,
        'issues': [],
        'recommendations': []
    }
    
    # Check for HSTS
    if 'strict-transport-security' in response_headers:
        results['score'] += 15
    else:
        results['issues'].append('Missing HSTS header')
        results['recommendations'].append('Add Strict-Transport-Security header')
    
    # Check for CSP
    if 'content-security-policy' in response_headers:
        results['score'] += 20
        csp = response_headers['content-security-policy'].lower()
        if "'unsafe-inline'" in csp or "'unsafe-eval'" in csp:
            results['issues'].append('CSP allows unsafe inline/eval')
            results['recommendations'].append('Remove unsafe-inline and unsafe-eval from CSP')
    else:
        results['issues'].append('Missing Content Security Policy')
        results['recommendations'].append('Add Content-Security-Policy header')
    
    # Check X-Frame-Options
    if 'x-frame-options' in response_headers:
        results['score'] += 10
    else:
        results['issues'].append('Missing X-Frame-Options header')
        results['recommendations'].append('Add X-Frame-Options: DENY')
    
    # Check X-Content-Type-Options
    if 'x-content-type-options' in response_headers:
        results['score'] += 10
    else:
        results['issues'].append('Missing X-Content-Type-Options header')
        results['recommendations'].append('Add X-Content-Type-Options: nosniff')
    
    # Check X-XSS-Protection
    if 'x-xss-protection' in response_headers:
        results['score'] += 5
    else:
        results['issues'].append('Missing X-XSS-Protection header')
        results['recommendations'].append('Add X-XSS-Protection: 1; mode=block')
    
    # Check Referrer-Policy
    if 'referrer-policy' in response_headers:
        results['score'] += 10
    else:
        results['issues'].append('Missing Referrer-Policy header')
        results['recommendations'].append('Add Referrer-Policy header')
    
    # Check Permissions-Policy
    if 'permissions-policy' in response_headers:
        results['score'] += 10
    else:
        results['issues'].append('Missing Permissions-Policy header')
        results['recommendations'].append('Add Permissions-Policy header')
    
    # Check for server information disclosure
    if 'server' in response_headers or 'x-powered-by' in response_headers:
        results['issues'].append('Server information disclosure')
        results['recommendations'].append('Remove Server and X-Powered-By headers')
    else:
        results['score'] += 10
    
    # Check cache control for sensitive content
    if 'cache-control' in response_headers:
        cache_control = response_headers['cache-control'].lower()
        if 'no-store' in cache_control or 'no-cache' in cache_control:
            results['score'] += 10
        else:
            results['recommendations'].append('Consider adding cache-control: no-store for sensitive content')
    
    # Grade the security
    if results['score'] >= 90:
        results['grade'] = 'A'
    elif results['score'] >= 80:
        results['grade'] = 'B'
    elif results['score'] >= 70:
        results['grade'] = 'C'
    elif results['score'] >= 60:
        results['grade'] = 'D'
    else:
        results['grade'] = 'F'
    
    return results


class CSPReportHandler:
    """Handler for Content Security Policy violation reports."""
    
    def __init__(self):
        self.violations: list = []
        self.max_violations = 1000
    
    def handle_violation(self, report: Dict[str, Any], client_ip: str):
        """Handle CSP violation report."""
        violation = {
            'timestamp': time.time(),
            'client_ip': client_ip,
            'document_uri': report.get('document-uri'),
            'violated_directive': report.get('violated-directive'),
            'blocked_uri': report.get('blocked-uri'),
            'source_file': report.get('source-file'),
            'line_number': report.get('line-number'),
            'column_number': report.get('column-number'),
            'original_policy': report.get('original-policy')
        }
        
        self.violations.append(violation)
        
        # Keep only recent violations
        if len(self.violations) > self.max_violations:
            self.violations = self.violations[-self.max_violations//2:]
        
        # Log violation
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"CSP Violation: {violation['violated_directive']} - {violation['blocked_uri']} from {client_ip}")
    
    def get_violations(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent CSP violations."""
        return sorted(self.violations, key=lambda x: x['timestamp'], reverse=True)[:limit]
    
    def get_violation_stats(self) -> Dict[str, Any]:
        """Get CSP violation statistics."""
        if not self.violations:
            return {'total_violations': 0}
        
        # Group by directive
        directive_counts = {}
        blocked_uri_counts = {}
        
        for violation in self.violations:
            directive = violation['violated_directive']
            directive_counts[directive] = directive_counts.get(directive, 0) + 1
            
            blocked_uri = violation['blocked_uri']
            blocked_uri_counts[blocked_uri] = blocked_uri_counts.get(blocked_uri, 0) + 1
        
        return {
            'total_violations': len(self.violations),
            'top_violated_directives': sorted(directive_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            'top_blocked_uris': sorted(blocked_uri_counts.items(), key=lambda x: x[1], reverse=True)[:5],
            'recent_violations': self.get_violations(10)
        }


# Global CSP report handler
_csp_handler = CSPReportHandler()


def get_csp_handler() -> CSPReportHandler:
    """Get global CSP report handler."""
    return _csp_handler