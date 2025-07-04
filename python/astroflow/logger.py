import sys
from loguru import logger

# Remove the default handler to avoid duplicate logs in some environments.
logger.remove()

# Add a file handler to log messages to a file with rotation, retention, and compression.
logger.add(
    "astroflow.log",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    rotation="10 MB",
    retention="10 days",
    encoding="utf-8",
)

__all__ = ["logger"]
