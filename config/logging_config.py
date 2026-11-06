"""Structured logging configuration for SentimentLab AI."""
import logging
import sys
from typing import Optional


def setup_logger(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a structured logger."""
    logger = logging.getLogger(name or "sentimentlab")
    
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        
    return logger

