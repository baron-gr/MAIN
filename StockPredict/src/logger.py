import logging
from typing import Optional

def get_console_logger(name: Optional[str] = 'StockPredict') -> logging.Logger:
    
    # create logger if it doesn't exist
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        # create console handler with formatting
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        # ddd console handler to the logger
        logger.addHandler(console_handler)

    return logger