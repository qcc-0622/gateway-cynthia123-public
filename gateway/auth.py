from fastapi import Request, HTTPException
from gateway.config import GATEWAY_API_KEY


def verify_api_key(request: Request):
    key = request.headers.get("x-api-key") or ""
    auth = request.headers.get("authorization") or ""
    if auth.startswith("Bearer "):
        key = key or auth[7:]
    if not GATEWAY_API_KEY:
        raise HTTPException(500, "Gateway API key not configured")
    if key != GATEWAY_API_KEY:
        raise HTTPException(401, "Invalid API key")
