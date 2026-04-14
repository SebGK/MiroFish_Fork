"""
Fehlerantwort-Verarbeitungswerkzeug
"""

import json
from typing import Any

from flask import Response


def sanitize_error_payload(payload: Any, status_code: int, debug_mode: bool = False) -> Any:
    """Traceback-Details aus 5xx-JSON-Antworten im Nicht-Debug-Modus entfernen"""
    if debug_mode or status_code < 500 or not isinstance(payload, dict) or 'traceback' not in payload:
        return payload

    sanitized_payload = dict(payload)
    sanitized_payload.pop('traceback', None)
    return sanitized_payload


def sanitize_json_error_response(response: Response, debug_mode: bool = False) -> Response:
    """Einheitliche Bereinigung von Flask-JSON-Antworten, um das Durchsickern interner Stack-Informationen an den Client zu vermeiden"""
    if not response.is_json:
        return response

    payload = response.get_json(silent=True)
    sanitized_payload = sanitize_error_payload(
        payload,
        status_code=response.status_code,
        debug_mode=debug_mode
    )

    if sanitized_payload is payload:
        return response

    response.set_data(json.dumps(sanitized_payload, ensure_ascii=False))
    return response
