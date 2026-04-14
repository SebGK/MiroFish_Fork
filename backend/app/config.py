"""
Konfigurationsverwaltung
Einheitliches Laden der Konfiguration aus der .env-Datei im Projektstammverzeichnis
"""

import os
import secrets
from dotenv import load_dotenv

# .env-Datei aus dem Projektstammverzeichnis laden
# Pfad: MiroFish/.env (relativ zu backend/app/config.py)
project_root_env = os.path.join(os.path.dirname(__file__), '../../.env')

if os.path.exists(project_root_env):
    load_dotenv(project_root_env, override=True)
else:
    # Falls keine .env im Stammverzeichnis vorhanden, Umgebungsvariablen laden (für Produktionsumgebung)
    load_dotenv(override=True)


def get_secret_key(environ=None) -> str:
    """Sicheren SECRET_KEY abrufen; bei fehlender Konfiguration wird ein prozessspezifischer Zufallswert generiert (ändert sich nach Neustart)"""
    env = os.environ if environ is None else environ
    return env.get('SECRET_KEY') or secrets.token_hex(32)


def get_debug_mode(environ=None) -> bool:
    """Debug-Modus-Umgebungsvariable parsen, standardmäßig deaktiviert; unterstützt 1/true/yes/on"""
    env = os.environ if environ is None else environ
    value = env.get('FLASK_DEBUG')
    if value is None:
        return False
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}


class Config:
    """Flask-Konfigurationsklasse"""
    
    # Flask-Konfiguration
    SECRET_KEY = get_secret_key()
    DEBUG = get_debug_mode()
    
    # JSON-Konfiguration - ASCII-Escaping deaktivieren, damit Nicht-ASCII-Zeichen direkt angezeigt werden (statt \uXXXX-Format)
    JSON_AS_ASCII = False
    
    # LLM-Konfiguration (einheitlich im OpenAI-Format)
    LLM_API_KEY = os.environ.get('LLM_API_KEY')
    LLM_BASE_URL = os.environ.get('LLM_BASE_URL', 'https://api.z.ai/api/coding/paas/v4')
    LLM_MODEL_NAME = os.environ.get('LLM_MODEL_NAME', 'glm-4.7')
    
    # Zep-Konfiguration
    ZEP_API_KEY = os.environ.get('ZEP_API_KEY')
    
    # Datei-Upload-Konfiguration
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024  # 50MB
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), '../uploads')
    ALLOWED_EXTENSIONS = {'pdf', 'md', 'txt', 'markdown'}
    
    # Textverarbeitungs-Konfiguration
    DEFAULT_CHUNK_SIZE = 500  # Standard-Chunk-Größe
    DEFAULT_CHUNK_OVERLAP = 50  # Standard-Überlappungsgröße
    
    # OASIS-Simulationskonfiguration
    OASIS_DEFAULT_MAX_ROUNDS = int(os.environ.get('OASIS_DEFAULT_MAX_ROUNDS', '10'))
    OASIS_SIMULATION_DATA_DIR = os.path.join(os.path.dirname(__file__), '../uploads/simulations')
    
    # OASIS-Plattform verfügbare Aktionen-Konfiguration
    OASIS_TWITTER_ACTIONS = [
        'CREATE_POST', 'LIKE_POST', 'REPOST', 'FOLLOW', 'DO_NOTHING', 'QUOTE_POST'
    ]
    OASIS_REDDIT_ACTIONS = [
        'LIKE_POST', 'DISLIKE_POST', 'CREATE_POST', 'CREATE_COMMENT',
        'LIKE_COMMENT', 'DISLIKE_COMMENT', 'SEARCH_POSTS', 'SEARCH_USER',
        'TREND', 'REFRESH', 'DO_NOTHING', 'FOLLOW', 'MUTE'
    ]
    
    # Report Agent配置
    REPORT_AGENT_MAX_TOOL_CALLS = int(os.environ.get('REPORT_AGENT_MAX_TOOL_CALLS', '5'))
    REPORT_AGENT_MAX_REFLECTION_ROUNDS = int(os.environ.get('REPORT_AGENT_MAX_REFLECTION_ROUNDS', '2'))
    REPORT_AGENT_TEMPERATURE = float(os.environ.get('REPORT_AGENT_TEMPERATURE', '0.5'))
    
    @classmethod
    def validate(cls):
        """Erforderliche Konfiguration validieren"""
        errors = []
        if not cls.LLM_API_KEY:
            errors.append("LLM_API_KEY nicht konfiguriert")
        if not cls.ZEP_API_KEY:
            errors.append("ZEP_API_KEY nicht konfiguriert")
        return errors
