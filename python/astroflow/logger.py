import sys
from loguru import logger
from .config.taskconfig import TaskConfig
import logging
logging.getLogger("ultralytics").setLevel(logging.ERROR)

# Remove the default handler to avoid duplicate logs in some environments.
logger.remove()

__all__ = ["logger"]
