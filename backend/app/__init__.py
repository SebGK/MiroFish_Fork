"""
MiroFish Backend - Flask-Anwendungsfabrik
"""

import os
import warnings

# Warnungen des multiprocessing resource_tracker unterdrücken (von Drittanbieter-Bibliotheken wie transformers)
# Muss vor allen anderen Importen gesetzt werden
warnings.filterwarnings("ignore", message=".*resource_tracker.*")

from flask import Flask, request
from flask_cors import CORS

from .config import Config
from .utils.logger import setup_logger, get_logger
from .utils.error_response import sanitize_json_error_response


def create_app(config_class=Config):
    """Flask-Anwendungsfabrik-Funktion"""
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # JSON-Kodierung einrichten: Sicherstellen, dass Nicht-ASCII-Zeichen direkt angezeigt werden (statt \uXXXX-Format)
    # Flask >= 2.3 verwendet app.json.ensure_ascii, ältere Versionen verwenden JSON_AS_ASCII-Konfiguration
    if hasattr(app, 'json') and hasattr(app.json, 'ensure_ascii'):
        app.json.ensure_ascii = False
    
    # Logging einrichten
    logger = setup_logger('mirofish')
    
    # Startinformationen nur im Reloader-Unterprozess ausgeben (um doppelte Ausgabe im Debug-Modus zu vermeiden)
    is_reloader_process = os.environ.get('WERKZEUG_RUN_MAIN') == 'true'
    debug_mode = app.config.get('DEBUG', False)
    should_log_startup = not debug_mode or is_reloader_process
    
    if should_log_startup:
        logger.info("=" * 50)
        logger.info("MiroFish Backend wird gestartet...")
        logger.info("=" * 50)
    
    # CORS aktivieren
    CORS(app, resources={r"/api/*": {"origins": "*"}})
    
    # Simulationsprozess-Bereinigungsfunktion registrieren (sicherstellen, dass alle Simulationsprozesse beim Herunterfahren beendet werden)
    from .services.simulation_runner import SimulationRunner
    SimulationRunner.register_cleanup()
    if should_log_startup:
        logger.info("Simulationsprozess-Bereinigungsfunktion registriert")
    
    # Anfrage-Logging-Middleware
    @app.before_request
    def log_request():
        logger = get_logger('mirofish.request')
        logger.debug(f"Anfrage: {request.method} {request.path}")
        if request.content_type and 'json' in request.content_type:
            logger.debug(f"Anfragekörper: {request.get_json(silent=True)}")
    
    @app.after_request
    def log_response(response):
        logger = get_logger('mirofish.request')
        logger.debug(f"Antwort: {response.status_code}")
        return sanitize_json_error_response(response, debug_mode=app.debug)
    
    # Blueprints registrieren
    from .api import graph_bp, simulation_bp, report_bp
    app.register_blueprint(graph_bp, url_prefix='/api/graph')
    app.register_blueprint(simulation_bp, url_prefix='/api/simulation')
    app.register_blueprint(report_bp, url_prefix='/api/report')
    
    # Gesundheitsprüfung
    @app.route('/health')
    def health():
        return {'status': 'ok', 'service': 'MiroFish Backend'}
    
    if should_log_startup:
        logger.info("MiroFish Backend erfolgreich gestartet")
    
    return app
