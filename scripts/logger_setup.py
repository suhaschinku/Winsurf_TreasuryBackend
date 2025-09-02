import logging
import os
from datetime import datetime, date
from contextlib import contextmanager
from functools import lru_cache
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def setup_logger(log_dir='logs', log_file='treasury_analysis.log', level=logging.INFO):
    """
    Set up and configure a logger instance with rotating file handler.
    
    Args:
        log_dir (str): Directory to store log files
        log_file (str): Name of the log file
        level (int): Logging level (e.g., logging.INFO, logging.DEBUG)
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Full path to log file
    log_path = os.path.join(log_dir, log_file)
    
    # Create logger
    logger = logging.getLogger('treasury_analysis')
    logger.setLevel(level)
    
    # Clear existing handlers if any
    if logger.handlers:
        logger.handlers = []
    
    # Create rotating file handler (max 10MB per file, keep 5 backup files)
    file_handler = RotatingFileHandler(
        log_path, maxBytes=10*1024*1024, backupCount=5
    )
    
    # Create console handler
    console_handler = logging.StreamHandler()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set formatter for handlers
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Global logger instance
logger = setup_logger()

def get_logger() -> logging.Logger:
    """
    Get the global logger instance.
    
    Returns:
        logging.Logger: The configured logger instance
    """
    return logger

@lru_cache(maxsize=128)
def read_template(filepath: str) -> str:
    """Cached template file reading"""
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        logger.error(f"Failed to read template {filepath}: {str(e)}")
        raise

def json_serializer(obj):
    """Custom JSON serializer for datetime objects"""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return str(obj)