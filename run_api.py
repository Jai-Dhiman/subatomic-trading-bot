#!/usr/bin/env python
"""
Startup script for Energy Trading API.

Usage:
    python run_api.py
"""

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
