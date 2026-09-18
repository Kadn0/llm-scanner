"""Shared download error handling independent of the Gradio application."""

import re

import requests


def response_error(response, operation):
    """Return the server's useful error for a failed download response."""
    detail = ""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            detail = str(payload.get("error") or payload.get("message") or "")
        elif payload:
            detail = str(payload)
    except (ValueError, requests.exceptions.JSONDecodeError):
        detail = (response.text or "").strip()
    detail = re.sub(r"\s+", " ", detail).strip()[:500]
    status = f"HTTP {response.status_code}"
    return f"{operation} failed ({status}): {detail}" if detail else f"{operation} failed ({status})"


def exception_message(exc, operation):
    """Format a download exception, retaining HTTP response details when available."""
    response = getattr(exc, "response", None)
    if response is not None and getattr(response, "status_code", 0):
        return response_error(response, operation)
    return f"{operation} failed: {exc}"


def is_permanent_error(exc):
    """Only retry transport failures and server errors; 4xx means the request cannot succeed unchanged."""
    response = getattr(exc, "response", None)
    return response is not None and 400 <= response.status_code < 500
