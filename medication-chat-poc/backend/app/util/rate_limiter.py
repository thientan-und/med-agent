# Rate Limiter for Medical AI API
# Implements token bucket and sliding window rate limiting

import time
import asyncio
from typing import Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque

from app.util.config import get_settings

settings = get_settings()


@dataclass
class RateLimitInfo:
    """Rate limit information for a client"""
    tokens: float = field(default=60.0)  # Available tokens
    last_refill: float = field(default_factory=time.time)  # Last token refill time
    requests: deque = field(default_factory=deque)  # Request timestamps


class RateLimiter:
    """Token bucket + sliding window rate limiter for medical API"""

    def __init__(self):
        self.clients: Dict[str, RateLimitInfo] = defaultdict(RateLimitInfo)
        self.max_requests_per_minute = settings.max_requests_per_minute
        self.max_requests_per_hour = settings.max_requests_per_hour

        # Token bucket parameters
        self.bucket_capacity = self.max_requests_per_minute
        self.refill_rate = self.max_requests_per_minute / 60.0  # tokens per second

    async def check_rate_limit(self, client_ip: str) -> bool:
        """
        Check if client is within rate limits

        Args:
            client_ip: Client IP address

        Returns:
            bool: True if request is allowed, False if rate limited
        """

        current_time = time.time()
        client_info = self.clients[client_ip]

        # 1. Token bucket algorithm (for burst control)
        if not await self._check_token_bucket(client_info, current_time):
            return False

        # 2. Sliding window algorithm (for sustained rate control)
        if not await self._check_sliding_window(client_info, current_time):
            return False

        # 3. Consume token and record request
        client_info.tokens -= 1
        client_info.requests.append(current_time)

        return True

    async def _check_token_bucket(self, client_info: RateLimitInfo, current_time: float) -> bool:
        """Check token bucket rate limit"""

        # Refill tokens based on time passed
        time_passed = current_time - client_info.last_refill
        tokens_to_add = time_passed * self.refill_rate

        client_info.tokens = min(
            self.bucket_capacity,
            client_info.tokens + tokens_to_add
        )
        client_info.last_refill = current_time

        # Check if we have tokens available
        return client_info.tokens >= 1.0

    async def _check_sliding_window(self, client_info: RateLimitInfo, current_time: float) -> bool:
        """Check sliding window rate limits"""

        # Remove old requests outside the time windows
        minute_ago = current_time - 60
        hour_ago = current_time - 3600

        # Remove requests older than 1 hour
        while client_info.requests and client_info.requests[0] < hour_ago:
            client_info.requests.popleft()

        # Count requests in last minute and hour
        requests_last_minute = sum(1 for req_time in client_info.requests if req_time > minute_ago)
        requests_last_hour = len(client_info.requests)

        # Check limits
        if requests_last_minute >= self.max_requests_per_minute:
            return False

        if requests_last_hour >= self.max_requests_per_hour:
            return False

        return True

    async def get_rate_limit_info(self, client_ip: str) -> Dict[str, any]:
        """Get current rate limit status for client"""

        current_time = time.time()
        client_info = self.clients[client_ip]

        # Update token bucket
        await self._check_token_bucket(client_info, current_time)

        # Count recent requests
        minute_ago = current_time - 60
        hour_ago = current_time - 3600

        requests_last_minute = sum(1 for req_time in client_info.requests if req_time > minute_ago)
        requests_last_hour = sum(1 for req_time in client_info.requests if req_time > hour_ago)

        return {
            "client_ip": client_ip,
            "tokens_available": int(client_info.tokens),
            "max_tokens": self.bucket_capacity,
            "requests_last_minute": requests_last_minute,
            "max_requests_per_minute": self.max_requests_per_minute,
            "requests_last_hour": requests_last_hour,
            "max_requests_per_hour": self.max_requests_per_hour,
            "rate_limited": (
                client_info.tokens < 1.0 or
                requests_last_minute >= self.max_requests_per_minute or
                requests_last_hour >= self.max_requests_per_hour
            )
        }

    async def reset_client_limits(self, client_ip: str):
        """Reset rate limits for a specific client (admin function)"""

        if client_ip in self.clients:
            del self.clients[client_ip]

    async def cleanup_old_entries(self):
        """Cleanup old client entries to prevent memory leaks"""

        current_time = time.time()
        hour_ago = current_time - 3600

        # Find clients with no recent activity
        inactive_clients = []
        for client_ip, client_info in self.clients.items():
            if not client_info.requests or client_info.requests[-1] < hour_ago:
                inactive_clients.append(client_ip)

        # Remove inactive clients
        for client_ip in inactive_clients:
            del self.clients[client_ip]

    async def get_system_stats(self) -> Dict[str, any]:
        """Get system-wide rate limiting statistics"""

        current_time = time.time()
        minute_ago = current_time - 60
        hour_ago = current_time - 3600

        total_clients = len(self.clients)
        active_clients_minute = 0
        active_clients_hour = 0
        total_requests_minute = 0
        total_requests_hour = 0
        rate_limited_clients = 0

        for client_info in self.clients.values():
            # Count requests
            requests_minute = sum(1 for req_time in client_info.requests if req_time > minute_ago)
            requests_hour = sum(1 for req_time in client_info.requests if req_time > hour_ago)

            total_requests_minute += requests_minute
            total_requests_hour += requests_hour

            if requests_minute > 0:
                active_clients_minute += 1
            if requests_hour > 0:
                active_clients_hour += 1

            # Check if rate limited
            if (client_info.tokens < 1.0 or
                requests_minute >= self.max_requests_per_minute or
                requests_hour >= self.max_requests_per_hour):
                rate_limited_clients += 1

        return {
            "total_clients_tracked": total_clients,
            "active_clients_last_minute": active_clients_minute,
            "active_clients_last_hour": active_clients_hour,
            "total_requests_last_minute": total_requests_minute,
            "total_requests_last_hour": total_requests_hour,
            "rate_limited_clients": rate_limited_clients,
            "max_requests_per_minute": self.max_requests_per_minute,
            "max_requests_per_hour": self.max_requests_per_hour
        }


# Global rate limiter instance
_rate_limiter = None


def get_rate_limiter() -> RateLimiter:
    """Get singleton rate limiter instance"""
    global _rate_limiter
    if _rate_limiter is None:
        _rate_limiter = RateLimiter()
    return _rate_limiter


# Background cleanup task
class RateLimiterCleanup:
    """Background task to cleanup old rate limiter entries"""

    def __init__(self, rate_limiter: RateLimiter, cleanup_interval: int = 3600):
        self.rate_limiter = rate_limiter
        self.cleanup_interval = cleanup_interval  # seconds
        self._task = None
        self._running = False

    async def start(self):
        """Start background cleanup task"""
        if self._running:
            return

        self._running = True
        self._task = asyncio.create_task(self._cleanup_loop())

    async def stop(self):
        """Stop background cleanup task"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while self._running:
            try:
                await self.rate_limiter.cleanup_old_entries()
                await asyncio.sleep(self.cleanup_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                # Log error but continue
                print(f"Rate limiter cleanup error: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying


# Decorator for rate limiting endpoints
def rate_limit(max_requests: Optional[int] = None):
    """
    Decorator to apply rate limiting to FastAPI endpoints

    Args:
        max_requests: Override default max requests per minute
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            from fastapi import Request, HTTPException

            # Find request object in arguments
            request = None
            for arg in args:
                if isinstance(arg, Request):
                    request = arg
                    break

            if not request:
                # If no request found, proceed without rate limiting
                return await func(*args, **kwargs)

            # Get client IP
            client_ip = request.client.host if request.client else "unknown"

            # Check rate limit
            rate_limiter = get_rate_limiter()
            if not await rate_limiter.check_rate_limit(client_ip):
                raise HTTPException(
                    status_code=429,
                    detail="Rate limit exceeded. Please try again later."
                )

            # Proceed with original function
            return await func(*args, **kwargs)

        return wrapper
    return decorator


# Export main components
__all__ = [
    "RateLimiter",
    "RateLimitInfo",
    "RateLimiterCleanup",
    "get_rate_limiter",
    "rate_limit"
]