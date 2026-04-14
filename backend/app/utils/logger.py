"""
Logging-Konfigurationsmodul
Bietet einheitliche Logging-Verwaltung mit gleichzeitiger Ausgabe auf Konsole und in Datei
"""

import os
import sys
import logging
from datetime import datetime
from logging.handlers import RotatingFileHandler


def _ensure_utf8_stdout():
    """
    Sicherstellen, dass stdout/stderr UTF-8-Kodierung verwenden
    Behebt Zeichenkodierungsprobleme in der Windows-Konsole
    """
    if sys.platform == 'win32':
        # Unter Windows Standardausgabe auf UTF-8 umkonfigurieren
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8', errors='replace')


# Log-Verzeichnis
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')


def setup_logger(name: str = 'mirofish', level: int = logging.DEBUG) -> logging.Logger:
    """
    Logger einrichten
    
    Args:
        name: Logger-Name
        level: Log-Level
        
    Returns:
        Konfigurierter Logger
    """
    # Sicherstellen, dass das Log-Verzeichnis existiert
    os.makedirs(LOG_DIR, exist_ok=True)
    
    # Logger erstellen
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Verhindern, dass Logs zum Root-Logger propagiert werden, um doppelte Ausgabe zu vermeiden
    logger.propagate = False
    
    # Wenn bereits Handler vorhanden sind, nicht erneut hinzufügen
    if logger.handlers:
        return logger
    
    # Log-Format
    detailed_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # 1. Datei-Handler - Detailliertes Logging (nach Datum benannt, mit Rotation)
    log_filename = datetime.now().strftime('%Y-%m-%d') + '.log'
    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR, log_filename),
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(detailed_formatter)
    
    # 2. Konsolen-Handler - Kompaktes Logging (INFO und höher)
    # Sicherstellen, dass unter Windows UTF-8-Kodierung verwendet wird, um Zeichenkodierungsprobleme zu vermeiden
    _ensure_utf8_stdout()
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    
    # Handler hinzufügen
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_logger(name: str = 'mirofish') -> logging.Logger:
    """
    Logger abrufen (falls nicht vorhanden, wird einer erstellt)
    
    Args:
        name: Logger-Name
        
    Returns:
        Logger-Instanz
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


# Standard-Logger erstellen
logger = setup_logger()


# Hilfsmethoden
def debug(msg, *args, **kwargs):
    logger.debug(msg, *args, **kwargs)

def info(msg, *args, **kwargs):
    logger.info(msg, *args, **kwargs)

def warning(msg, *args, **kwargs):
    logger.warning(msg, *args, **kwargs)

def error(msg, *args, **kwargs):
    logger.error(msg, *args, **kwargs)

def critical(msg, *args, **kwargs):
    logger.critical(msg, *args, **kwargs)

