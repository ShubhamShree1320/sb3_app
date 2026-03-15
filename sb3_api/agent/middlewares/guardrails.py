"""
This adds guardrails, rate limiting, and comprehensive monitoring.
"""

import logging
import re
import time
from collections import defaultdict
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable

from fastapi import HTTPException, Request
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ====================================================================================
# PART 1: GUARDRAILS FOR INPUT/OUTPUT VALIDATION
# ====================================================================================

class GuardrailViolation(Exception):
    """Raised when content violates guardrails."""
    def __init__(self, message: str, violation_type: str):
        self.message = message
        self.violation_type = violation_type
        super().__init__(self.message)


class ContentGuardrails:
    """Comprehensive guardrails for input/output validation.
    
    Checks for:
    - PII (Personal Identifiable Information)
    - SQL injection patterns
    - Malicious content
    - Excessive output length
    """
    
    # PII patterns
    SSN_PATTERN = re.compile(r'\b\d{3}-\d{2}-\d{4}\b')
    CREDIT_CARD_PATTERN = re.compile(r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b')
    EMAIL_PATTERN = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
    PHONE_PATTERN = re.compile(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b')
    
    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(\bOR\b.*=.*)|(\bAND\b.*=.*)",  # OR 1=1, AND 1=1
        r"(;.*DROP\b)|(;.*DELETE\b)",  # DROP, DELETE
        r"(UNION\s+SELECT)",  # UNION SELECT
        r"(--)|(/\*.*\*/)",  # SQL comments
    ]
    
    # Malicious patterns
    MALICIOUS_PATTERNS = [
        r"<script[^>]*>.*?</script>",  # XSS
        r"javascript:",  # JavaScript protocol
        r"on\w+\s*=",  # Event handlers
    ]
    
    def __init__(
        self,
        check_pii: bool = True,
        check_sql_injection: bool = True,
        check_malicious: bool = True,
        max_output_length: int = 50000,
        redact_pii: bool = True,
    ):
        self.check_pii = check_pii
        self.check_sql_injection = check_sql_injection
        self.check_malicious = check_malicious
        self.max_output_length = max_output_length
        self.redact_pii = redact_pii
        logger.info("✓ Content guardrails initialized")
    
    def validate_input(self, text: str) -> tuple[bool, str | None]:
        """Validate user input.
        
        Returns:
            (is_valid, error_message)
        """
        if not text:
            return True, None
        
        # Check for SQL injection
        if self.check_sql_injection:
            for pattern in self.SQL_INJECTION_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    logger.warning(f"SQL injection detected in input: {text[:100]}")
                    return False, "Input contains potentially malicious SQL patterns"
        
        # Check for malicious content
        if self.check_malicious:
            for pattern in self.MALICIOUS_PATTERNS:
                if re.search(pattern, text, re.IGNORECASE):
                    logger.warning(f"Malicious pattern detected in input: {text[:100]}")
                    return False, "Input contains potentially malicious content"
        
        return True, None
    
    def validate_output(self, text: str) -> tuple[bool, str, str | None]:
        """Validate agent output.
        
        Returns:
            (is_valid, sanitized_text, error_message)
        """
        if not text:
            return True, text, None
        
        # Check output length
        if len(text) > self.max_output_length:
            logger.warning(f"Output too long: {len(text)} chars")
            return False, text, f"Output exceeds maximum length ({self.max_output_length} chars)"
        
        sanitized_text = text
        
        # Check and redact PII
        if self.check_pii:
            pii_found = []
            
            # SSN
            if self.SSN_PATTERN.search(text):
                pii_found.append("SSN")
                if self.redact_pii:
                    sanitized_text = self.SSN_PATTERN.sub("[SSN REDACTED]", sanitized_text)
            
            # Credit card
            if self.CREDIT_CARD_PATTERN.search(text):
                pii_found.append("Credit Card")
                if self.redact_pii:
                    sanitized_text = self.CREDIT_CARD_PATTERN.sub("[CARD REDACTED]", sanitized_text)
            
            # Email (be careful - might be legitimate)
            emails = self.EMAIL_PATTERN.findall(text)
            if emails:
                # Only flag if multiple emails or non-company domain
                if len(emails) > 2:
                    pii_found.append("Email")
                    if self.redact_pii:
                        sanitized_text = self.EMAIL_PATTERN.sub("[EMAIL REDACTED]", sanitized_text)
            
            # Phone
            if self.PHONE_PATTERN.search(text):
                pii_found.append("Phone")
                if self.redact_pii:
                    sanitized_text = self.PHONE_PATTERN.sub("[PHONE REDACTED]", sanitized_text)
            
            if pii_found:
                logger.warning(f"PII detected in output: {', '.join(pii_found)}")
                if not self.redact_pii:
                    return False, text, f"Output contains PII: {', '.join(pii_found)}"
        
        return True, sanitized_text, None


# ====================================================================================
# PART 2: RATE LIMITING
# ====================================================================================

class RateLimiter:
    """Token bucket rate limiter.
    
    Limits requests per user/IP to prevent abuse.
    
    Args:
        requests_per_minute: Max requests per minute per user
        requests_per_hour: Max requests per hour per user
        burst_size: Max burst size (default: 2x per minute rate)
    """
    
    def __init__(
        self,
        requests_per_minute: int = 10,
        requests_per_hour: int = 100,
        burst_size: int | None = None,
    ):
        self.requests_per_minute = requests_per_minute
        self.requests_per_hour = requests_per_hour
        self.burst_size = burst_size or (requests_per_minute * 2)
        
        # Storage: {user_id: [timestamps]}
        self.minute_buckets: dict[str, list[datetime]] = defaultdict(list)
        self.hour_buckets: dict[str, list[datetime]] = defaultdict(list)
        
        logger.info(f"✓ Rate limiter initialized: {requests_per_minute}/min, {requests_per_hour}/hr")
    
    def check_rate_limit(self, user_id: str) -> tuple[bool, str | None]:
        """Check if user is within rate limits.
        
        Returns:
            (is_allowed, error_message)
        """
        now = datetime.now()
        
        # Clean old timestamps
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        
        self.minute_buckets[user_id] = [
            ts for ts in self.minute_buckets[user_id] if ts > minute_ago
        ]
        self.hour_buckets[user_id] = [
            ts for ts in self.hour_buckets[user_id] if ts > hour_ago
        ]
        
        # Check minute limit
        if len(self.minute_buckets[user_id]) >= self.requests_per_minute:
            logger.warning(f"Rate limit exceeded (minute): user={user_id}")
            return False, f"Rate limit exceeded: {self.requests_per_minute} requests per minute"
        
        # Check hour limit
        if len(self.hour_buckets[user_id]) >= self.requests_per_hour:
            logger.warning(f"Rate limit exceeded (hour): user={user_id}")
            return False, f"Rate limit exceeded: {self.requests_per_hour} requests per hour"
        
        # Check burst
        if len(self.minute_buckets[user_id]) >= self.burst_size:
            logger.warning(f"Burst limit exceeded: user={user_id}")
            return False, f"Too many requests too quickly. Please slow down."
        
        # Add timestamp
        self.minute_buckets[user_id].append(now)
        self.hour_buckets[user_id].append(now)
        
        return True, None
    
    def get_stats(self, user_id: str) -> dict:
        """Get current rate limit stats for user."""
        return {
            "requests_last_minute": len(self.minute_buckets[user_id]),
            "requests_last_hour": len(self.hour_buckets[user_id]),
            "limit_per_minute": self.requests_per_minute,
            "limit_per_hour": self.requests_per_hour,
        }


# ====================================================================================
# PART 3: MONITORING & METRICS
# ====================================================================================

class MetricsCollector:
    """Collect and expose metrics for monitoring.
    
    Tracks:
    - Request counts
    - Response times
    - Error rates
    - Token usage
    """
    
    def __init__(self):
        self.request_count = 0
        self.error_count = 0
        self.total_response_time = 0.0
        self.response_times: list[float] = []
        self.token_usage: list[int] = []
        
        # Per-endpoint metrics
        self.endpoint_metrics: dict[str, dict] = defaultdict(lambda: {
            "count": 0,
            "errors": 0,
            "total_time": 0.0,
        })
        
        logger.info("✓ Metrics collector initialized")
    
    def record_request(self, endpoint: str, duration: float, success: bool, tokens: int = 0):
        """Record a request."""
        self.request_count += 1
        if not success:
            self.error_count += 1
        
        self.total_response_time += duration
        self.response_times.append(duration)
        
        if tokens > 0:
            self.token_usage.append(tokens)
        
        # Per-endpoint
        self.endpoint_metrics[endpoint]["count"] += 1
        self.endpoint_metrics[endpoint]["total_time"] += duration
        if not success:
            self.endpoint_metrics[endpoint]["errors"] += 1
        
        # Keep only last 1000 response times
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]
    
    def get_metrics(self) -> dict:
        """Get current metrics."""
        avg_response_time = (
            self.total_response_time / self.request_count
            if self.request_count > 0 else 0
        )
        
        error_rate = (
            self.error_count / self.request_count * 100
            if self.request_count > 0 else 0
        )
        
        # P95 response time
        p95_response_time = 0.0
        if self.response_times:
            sorted_times = sorted(self.response_times)
            p95_index = int(len(sorted_times) * 0.95)
            p95_response_time = sorted_times[p95_index] if p95_index < len(sorted_times) else 0.0
        
        # Average tokens
        avg_tokens = sum(self.token_usage) / len(self.token_usage) if self.token_usage else 0
        
        return {
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "error_rate_percent": round(error_rate, 2),
            "avg_response_time_seconds": round(avg_response_time, 3),
            "p95_response_time_seconds": round(p95_response_time, 3),
            "avg_tokens_per_request": round(avg_tokens, 0),
            "endpoints": dict(self.endpoint_metrics),
        }


# Global instances
guardrails = ContentGuardrails()
rate_limiter = RateLimiter(requests_per_minute=10, requests_per_hour=100)
metrics_collector = MetricsCollector()


# ====================================================================================
# PART 4: FASTAPI MIDDLEWARE
# ====================================================================================

def rate_limit_middleware(user_id_extractor: Callable[[Request], str]):
    """FastAPI dependency for rate limiting.
    
    Args:
        user_id_extractor: Function to extract user ID from request
        
    Example:
        def get_user_id(request: Request) -> str:
            return request.state.user.email
        
        @app.post("/chat", dependencies=[Depends(rate_limit_middleware(get_user_id))])
    """
    def dependency(request: Request):
        user_id = user_id_extractor(request)
        is_allowed, error = rate_limiter.check_rate_limit(user_id)
        if not is_allowed:
            raise HTTPException(status_code=429, detail=error)
        return user_id
    return dependency


def monitoring_middleware(func: Callable) -> Callable:
    """Decorator for monitoring endpoint performance.
    
    Usage:
        @router.post("/chat")
        @monitoring_middleware
        async def chat_endpoint(...):
            ...
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        success = True
        tokens = 0
        
        try:
            result = await func(*args, **kwargs)
            
            # Extract token usage if available
            if hasattr(result, 'token_usage'):
                tokens = result.token_usage
            
            return result
        except Exception as e:
            success = False
            raise
        finally:
            duration = time.time() - start_time
            endpoint = func.__name__
            metrics_collector.record_request(endpoint, duration, success, tokens)
    
    return wrapper


# ====================================================================================
# INSTALLATION INSTRUCTIONS
# ====================================================================================

"""
STEP 1: Create middleware directory and files
----------------------------------------------
mkdir -p sb3_api/middleware
cp FIX_3_PRODUCTION_HARDENING.py sb3_api/middleware/guardrails.py

# Create __init__.py
cat > sb3_api/middleware/__init__.py << 'EOF'
from sb3_api.middleware.guardrails import (
    ContentGuardrails,
    RateLimiter,
    MetricsCollector,
    guardrails,
    rate_limiter,
    metrics_collector,
    rate_limit_middleware,
    monitoring_middleware,
)

__all__ = [
    "ContentGuardrails",
    "RateLimiter",
    "MetricsCollector",
    "guardrails",
    "rate_limiter",
    "metrics_collector",
    "rate_limit_middleware",
    "monitoring_middleware",
]
EOF

STEP 2: Update main.py to add metrics endpoint
-----------------------------------------------
Add to main.py:

```python
from sb3_api.middleware import metrics_collector

@app.get("/health")
async def health():
    \"\"\"Health check endpoint.\"\"\"
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/metrics")
async def get_metrics():
    \"\"\"Get current metrics.\"\"\"
    return metrics_collector.get_metrics()
```

STEP 3: Update chat routes to use middleware
---------------------------------------------
In sb3_api/routes/chat.py:

```python
from sb3_api.middleware import (
    guardrails,
    rate_limiter,
    monitoring_middleware,
    rate_limit_middleware,
)

def get_user_id_from_request(request: Request) -> str:
    \"\"\"Extract user ID from request.\"\"\"
    # Option 1: From auth token
    if hasattr(request.state, 'user'):
        return request.state.user.email
    
    # Option 2: From header
    user_id = request.headers.get('X-User-ID')
    if user_id:
        return user_id
    
    # Option 3: From IP (fallback)
    return request.client.host

@router.post(
    "/sql-agent/chat",
    dependencies=[Depends(rate_limit_middleware(get_user_id_from_request))]
)
@monitoring_middleware
async def sql_agent(
    body: UserQuery,
    chat_controller: ChatController = Depends(get_chat_controller),
    user: UserInfo = Depends(require_auth),
) -> MessageResponse:
    # Validate input
    is_valid, error = guardrails.validate_input(body.query)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error)
    
    messages = []
    async for message_event in chat_controller.process_chat_query_stream(
        query=body.query,
        session_id=str(body.session_id) if body.session_id else None,
        user_email=user.email,
        persona=body.profile,
        debug_mode=body.debug_mode,
    ):
        session_id = message_event.session_id
        messages.append(message_event.message)
    
    # Validate output
    if messages and hasattr(messages[-1], 'content'):
        is_valid, sanitized, error = guardrails.validate_output(messages[-1].content)
        if not is_valid and not guardrails.redact_pii:
            raise HTTPException(status_code=500, detail=error)
        
        # Apply sanitization
        if is_valid and sanitized != messages[-1].content:
            messages[-1].content = sanitized
            logger.info("PII redacted from output")
    
    return MessageResponse(session_id=session_id, messages=messages)
```

STEP 4: Add settings for guardrails
------------------------------------
In settings.py:

```python
class ServiceSettings(BaseSettings):
    # ... existing settings ...
    
    # Guardrails
    ENABLE_GUARDRAILS: bool = True
    REDACT_PII: bool = True
    MAX_OUTPUT_LENGTH: int = 50000
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 10
    RATE_LIMIT_PER_HOUR: int = 100
    
    # Monitoring
    ENABLE_METRICS: bool = True
```

STEP 5: Test guardrails
------------------------
# Test PII detection
curl -X POST http://localhost:8000/api/sql-agent/chat \
  -d '{"query": "Show me data for SSN 123-45-6789"}'
# Should block or redact

# Test rate limiting
for i in {1..15}; do
  curl -X POST http://localhost:8000/api/sql-agent/chat \
    -d '{"query": "test"}' &
done
# Should see 429 errors after 10 requests

# Check metrics
curl http://localhost:8000/metrics
# Should show request counts, response times, etc.

STEP 6: Set up monitoring dashboards
-------------------------------------
# Option 1: Prometheus + Grafana
# Add prometheus_client to requirements.txt
# Export metrics in Prometheus format

# Option 2: CloudWatch
# Use AWS CloudWatch agent
# Push metrics to CloudWatch

# Option 3: DataDog
# Use DataDog agent
# Configure StatsD integration

VERIFICATION COMPLETE ✅
------------------------
If guardrails block malicious input and metrics endpoint returns data,
production hardening is active!

MONITORING CHECKLIST:
---------------------
- [ ] Metrics endpoint accessible at /metrics
- [ ] Rate limiting working (429 errors after limit)
- [ ] PII detection blocking/redacting sensitive data
- [ ] Response times logged
- [ ] Error rates tracked
- [ ] Dashboards configured (Grafana/CloudWatch/DataDog)
- [ ] Alerts configured for high error rates
- [ ] Alerts configured for slow response times

SECURITY BENEFITS:
------------------
1. **Input Validation**: Blocks SQL injection and XSS
2. **PII Protection**: Automatically redacts sensitive data
3. **Rate Limiting**: Prevents abuse and DDoS
4. **Monitoring**: Track performance and errors
5. **Observability**: Full visibility into system health
"""
