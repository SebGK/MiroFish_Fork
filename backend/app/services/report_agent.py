"""
Report Agent-Dienst
Simulationsberichtserstellung im ReACT-Modus mit LangChain + Zep

Funktionen:
1. Berichterstellung basierend auf Simulationsanforderungen und Zep-Graphinformationen
2. Zunächst Gliederungsstruktur planen, dann abschnittsweise generieren
3. Jeder Abschnitt verwendet den ReACT-Modus mit mehreren Denk- und Reflexionsrunden
4. Unterstützung für Benutzerdialoge mit autonomem Abruf von Suchwerkzeugen
"""

import os
import json
import time
import re
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

from ..config import Config
from ..utils.llm_client import LLMClient
from ..utils.logger import get_logger
from ..utils.locale import get_language_instruction, t
from .zep_tools import (
    ZepToolsService, 
    SearchResult, 
    InsightForgeResult, 
    PanoramaResult,
    InterviewResult
)

logger = get_logger('mirofish.report_agent')


class ReportLogger:
    """
    Report Agent – Detaillierter Protokollierer
    
    Erzeugt eine agent_log.jsonl-Datei im Berichtsordner und protokolliert jeden Schritt im Detail.
    Jede Zeile ist ein vollständiges JSON-Objekt mit Zeitstempel, Aktionstyp, Detailinhalten usw.
    """
    
    def __init__(self, report_id: str):
        """
        Protokollierer initialisieren
        
        Args:
            report_id: Berichts-ID, zur Bestimmung des Protokolldateipfads
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'agent_log.jsonl'
        )
        self.start_time = datetime.now()
        self._ensure_log_file()
    
    def _ensure_log_file(self):
        """Sicherstellen, dass das Verzeichnis der Protokolldatei existiert"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)
    
    def _get_elapsed_time(self) -> float:
        """Verstrichene Zeit seit dem Start ermitteln (in Sekunden)"""
        return (datetime.now() - self.start_time).total_seconds()
    
    def log(
        self, 
        action: str, 
        stage: str,
        details: Dict[str, Any],
        section_title: str = None,
        section_index: int = None
    ):
        """
        Einen Protokolleintrag aufzeichnen
        
        Args:
            action: Aktionstyp, z.B. 'start', 'tool_call', 'llm_response', 'section_complete' usw.
            stage: Aktuelle Phase, z.B. 'planning', 'generating', 'completed'
            details: Detailinhalt-Dictionary, nicht abgeschnitten
            section_title: Aktueller Kapitelüberschrift (optional)
            section_index: Aktueller Kapitelindex (optional)
        """
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(self._get_elapsed_time(), 2),
            "report_id": self.report_id,
            "action": action,
            "stage": stage,
            "section_title": section_title,
            "section_index": section_index,
            "details": details
        }
        
        # An JSONL-Datei anhängen
        with open(self.log_file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    
    def log_start(self, simulation_id: str, graph_id: str, simulation_requirement: str):
        """Beginn der Berichtserstellung protokollieren"""
        self.log(
            action="report_start",
            stage="pending",
            details={
                "simulation_id": simulation_id,
                "graph_id": graph_id,
                "simulation_requirement": simulation_requirement,
                "message": t('report.taskStarted')
            }
        )
    
    def log_planning_start(self):
        """Beginn der Gliederungsplanung protokollieren"""
        self.log(
            action="planning_start",
            stage="planning",
            details={"message": t('report.planningStart')}
        )
    
    def log_planning_context(self, context: Dict[str, Any]):
        """Kontextinformationen bei der Planung protokollieren"""
        self.log(
            action="planning_context",
            stage="planning",
            details={
                "message": t('report.fetchSimContext'),
                "context": context
            }
        )
    
    def log_planning_complete(self, outline_dict: Dict[str, Any]):
        """Abschluss der Gliederungsplanung protokollieren"""
        self.log(
            action="planning_complete",
            stage="planning",
            details={
                "message": t('report.planningComplete'),
                "outline": outline_dict
            }
        )
    
    def log_section_start(self, section_title: str, section_index: int):
        """Beginn der Kapitelgenerierung protokollieren"""
        self.log(
            action="section_start",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={"message": t('report.sectionStart', title=section_title)}
        )
    
    def log_react_thought(self, section_title: str, section_index: int, iteration: int, thought: str):
        """ReACT-Denkprozess protokollieren"""
        self.log(
            action="react_thought",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "thought": thought,
                "message": t('report.reactThought', iteration=iteration)
            }
        )
    
    def log_tool_call(
        self, 
        section_title: str, 
        section_index: int,
        tool_name: str, 
        parameters: Dict[str, Any],
        iteration: int
    ):
        """Werkzeugaufruf protokollieren"""
        self.log(
            action="tool_call",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "parameters": parameters,
                "message": t('report.toolCall', toolName=tool_name)
            }
        )
    
    def log_tool_result(
        self,
        section_title: str,
        section_index: int,
        tool_name: str,
        result: str,
        iteration: int
    ):
        """Werkzeugaufruf-Ergebnis protokollieren (vollständiger Inhalt, nicht abgeschnitten)"""
        self.log(
            action="tool_result",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "tool_name": tool_name,
                "result": result,  # Vollständiges Ergebnis, nicht abgeschnitten
                "result_length": len(result),
                "message": t('report.toolResult', toolName=tool_name)
            }
        )
    
    def log_llm_response(
        self,
        section_title: str,
        section_index: int,
        response: str,
        iteration: int,
        has_tool_calls: bool,
        has_final_answer: bool
    ):
        """LLM-Antwort protokollieren (vollständiger Inhalt, nicht abgeschnitten)"""
        self.log(
            action="llm_response",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "iteration": iteration,
                "response": response,  # Vollständige Antwort, nicht abgeschnitten
                "response_length": len(response),
                "has_tool_calls": has_tool_calls,
                "has_final_answer": has_final_answer,
                "message": t('report.llmResponse', hasToolCalls=has_tool_calls, hasFinalAnswer=has_final_answer)
            }
        )
    
    def log_section_content(
        self,
        section_title: str,
        section_index: int,
        content: str,
        tool_calls_count: int
    ):
        """Kapitelinhalt-Generierung abgeschlossen protokollieren (nur Inhalt, bedeutet nicht, dass das gesamte Kapitel abgeschlossen ist)"""
        self.log(
            action="section_content",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": content,  # Vollständiger Inhalt, nicht abgeschnitten
                "content_length": len(content),
                "tool_calls_count": tool_calls_count,
                "message": t('report.sectionContentDone', title=section_title)
            }
        )
    
    def log_section_full_complete(
        self,
        section_title: str,
        section_index: int,
        full_content: str
    ):
        """
        Kapitelgenerierung abgeschlossen protokollieren

        Das Frontend sollte dieses Protokoll überwachen, um festzustellen, ob ein Kapitel wirklich abgeschlossen ist, und den vollständigen Inhalt zu erhalten
        """
        self.log(
            action="section_complete",
            stage="generating",
            section_title=section_title,
            section_index=section_index,
            details={
                "content": full_content,
                "content_length": len(full_content),
                "message": t('report.sectionComplete', title=section_title)
            }
        )
    
    def log_report_complete(self, total_sections: int, total_time_seconds: float):
        """Abschluss der Berichtserstellung protokollieren"""
        self.log(
            action="report_complete",
            stage="completed",
            details={
                "total_sections": total_sections,
                "total_time_seconds": round(total_time_seconds, 2),
                "message": t('report.reportComplete')
            }
        )
    
    def log_error(self, error_message: str, stage: str, section_title: str = None):
        """Fehler protokollieren"""
        self.log(
            action="error",
            stage=stage,
            section_title=section_title,
            section_index=None,
            details={
                "error": error_message,
                "message": t('report.errorOccurred', error=error_message)
            }
        )


class ReportConsoleLogger:
    """
    Report Agent – Konsolen-Protokollierer
    
    Schreibt konsolenformatige Protokolle (INFO, WARNING usw.) in die Datei console_log.txt im Berichtsordner.
    Diese Protokolle unterscheiden sich von agent_log.jsonl und sind reine Textausgaben im Konsolenformat.
    """
    
    def __init__(self, report_id: str):
        """
        Konsolen-Protokollierer initialisieren
        
        Args:
            report_id: Berichts-ID, zur Bestimmung des Protokolldateipfads
        """
        self.report_id = report_id
        self.log_file_path = os.path.join(
            Config.UPLOAD_FOLDER, 'reports', report_id, 'console_log.txt'
        )
        self._ensure_log_file()
        self._file_handler = None
        self._setup_file_handler()
    
    def _ensure_log_file(self):
        """Sicherstellen, dass das Verzeichnis der Protokolldatei existiert"""
        log_dir = os.path.dirname(self.log_file_path)
        os.makedirs(log_dir, exist_ok=True)
    
    def _setup_file_handler(self):
        """Datei-Handler einrichten, um Protokolle gleichzeitig in eine Datei zu schreiben"""
        import logging
        
        # Datei-Handler erstellen
        self._file_handler = logging.FileHandler(
            self.log_file_path,
            mode='a',
            encoding='utf-8'
        )
        self._file_handler.setLevel(logging.INFO)
        
        # Gleiches kompaktes Format wie die Konsole verwenden
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%H:%M:%S'
        )
        self._file_handler.setFormatter(formatter)
        
        # Zu report_agent-bezogenen Loggern hinzufügen
        loggers_to_attach = [
            'mirofish.report_agent',
            'mirofish.zep_tools',
        ]
        
        for logger_name in loggers_to_attach:
            target_logger = logging.getLogger(logger_name)
            # Doppeltes Hinzufügen vermeiden
            if self._file_handler not in target_logger.handlers:
                target_logger.addHandler(self._file_handler)
    
    def close(self):
        """Datei-Handler schließen und vom Logger entfernen"""
        import logging
        
        if self._file_handler:
            loggers_to_detach = [
                'mirofish.report_agent',
                'mirofish.zep_tools',
            ]
            
            for logger_name in loggers_to_detach:
                target_logger = logging.getLogger(logger_name)
                if self._file_handler in target_logger.handlers:
                    target_logger.removeHandler(self._file_handler)
            
            self._file_handler.close()
            self._file_handler = None
    
    def __del__(self):
        """Beim Destruktor sicherstellen, dass der Datei-Handler geschlossen wird"""
        self.close()


class ReportStatus(str, Enum):
    """Berichtsstatus"""
    PENDING = "pending"
    PLANNING = "planning"
    GENERATING = "generating"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ReportSection:
    """Berichtskapitel"""
    title: str
    content: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "content": self.content
        }

    def to_markdown(self, level: int = 2) -> str:
        """In Markdown-Format konvertieren"""
        if self.content:
            md += f"{self.content}\n\n"
        return md


@dataclass
class ReportOutline:
    """Berichtsgliederung"""
    title: str
    summary: str
    sections: List[ReportSection]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "summary": self.summary,
            "sections": [s.to_dict() for s in self.sections]
        }
    
    def to_markdown(self) -> str:
        """In Markdown-Format konvertieren"""
        md = f"# {self.title}\n\n"
        md += f"> {self.summary}\n\n"
        for section in self.sections:
            md += section.to_markdown()
        return md


@dataclass
class Report:
    """Vollständiger Bericht"""
    report_id: str
    simulation_id: str
    graph_id: str
    simulation_requirement: str
    status: ReportStatus
    outline: Optional[ReportOutline] = None
    markdown_content: str = ""
    created_at: str = ""
    completed_at: str = ""
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "report_id": self.report_id,
            "simulation_id": self.simulation_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "status": self.status.value,
            "outline": self.outline.to_dict() if self.outline else None,
            "markdown_content": self.markdown_content,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "error": self.error
        }


# ═══════════════════════════════════════════════════════════════
# Prompt-Vorlagenkonstanten
# ═══════════════════════════════════════════════════════════════

# ── Werkzeugbeschreibungen ──

TOOL_DESC_INSIGHT_FORGE = """\
【Tiefgehende Erkenntnissuche – Leistungsstarkes Suchwerkzeug】
Dies ist unsere leistungsstarke Suchfunktion, speziell für Tiefenanalysen entwickelt. Sie wird:
1. Ihre Frage automatisch in mehrere Teilfragen zerlegen
2. Informationen aus mehreren Dimensionen im Simulationsgraphen abrufen
3. Ergebnisse aus semantischer Suche, Entitätsanalyse und Beziehungskettenverfolgung integrieren
4. Die umfassendsten und tiefgreifendsten Suchergebnisse liefern

【Einsatzszenarien】
- Wenn ein Thema tiefgehend analysiert werden muss
- Wenn mehrere Aspekte eines Ereignisses verstanden werden müssen
- Wenn reichhaltiges Material zur Unterstützung von Berichtskapiteln benötigt wird

【Rückgabeinhalte】
- Relevante Originalfakten (direkt zitierbar)
- Kernentitätserkenntnisse
- Beziehungskettenanalyse"""

TOOL_DESC_PANORAMA_SEARCH = """\
【Breitbandsuche – Gesamtübersicht erhalten】
Dieses Werkzeug dient dazu, einen vollständigen Überblick über die Simulationsergebnisse zu erhalten, besonders geeignet für das Verständnis von Ereignisverläufen. Es wird:
1. Alle relevanten Knoten und Beziehungen abrufen
2. Zwischen aktuell gültigen Fakten und historischen/veralteten Fakten unterscheiden
3. Ihnen helfen zu verstehen, wie sich die öffentliche Meinung entwickelt hat

【Einsatzszenarien】
- Wenn der vollständige Entwicklungsverlauf eines Ereignisses verstanden werden muss
- Wenn Meinungsänderungen in verschiedenen Phasen verglichen werden müssen
- Wenn umfassende Entitäts- und Beziehungsinformationen benötigt werden

【Rückgabeinhalte】
- Aktuell gültige Fakten (neueste Simulationsergebnisse)
- Historische/veraltete Fakten (Entwicklungsprotokoll)
- Alle beteiligten Entitäten"""

TOOL_DESC_QUICK_SEARCH = """\
【Einfache Suche – Schnellsuche】
Leichtgewichtiges Schnellsuchwerkzeug, geeignet für einfache und direkte Informationsabfragen.

【Einsatzszenarien】
- Wenn eine bestimmte Information schnell gefunden werden muss
- Wenn ein Fakt verifiziert werden muss
- Einfache Informationssuche

【Rückgabeinhalte】
- Liste der zur Abfrage relevantesten Fakten"""

TOOL_DESC_INTERVIEW_AGENTS = """\
【Tiefeninterview – Echte Agent-Interviews (Dual-Plattform)】
Ruft die Interview-API der OASIS-Simulationsumgebung auf, um echte Interviews mit laufenden Simulations-Agents durchzuführen!
Dies ist keine LLM-Simulation, sondern ein Aufruf der echten Interview-Schnittstelle für Originalantworten der Simulations-Agents.
Standardmäßig werden Interviews auf Twitter und Reddit gleichzeitig durchgeführt, um umfassendere Perspektiven zu erhalten.

Funktionsablauf:
1. Automatisches Lesen der Persona-Datei, um alle Simulations-Agents kennenzulernen
2. Intelligente Auswahl der zum Interviewthema relevantesten Agents (z.B. Studenten, Medien, Behörden usw.)
3. Automatische Generierung von Interviewfragen
4. Aufruf der /api/simulation/interview/batch-Schnittstelle für echte Dual-Plattform-Interviews
5. Integration aller Interviewergebnisse mit Mehrperspektiven-Analyse

【Einsatzszenarien】
- Wenn Meinungen aus verschiedenen Rollenperspektiven benötigt werden (Was denken Studenten? Was sagen Medien? Was sagen Behörden?)
- Wenn Meinungen und Standpunkte verschiedener Seiten gesammelt werden müssen
- Wenn echte Antworten von Simulations-Agents benötigt werden (aus der OASIS-Simulationsumgebung)
- Wenn der Bericht lebendiger sein soll, mit „Interviewprotokollen"

【Rückgabeinhalte】
- Identitätsinformationen der interviewten Agents
- Interviewantworten der Agents auf Twitter und Reddit
- Schlüsselzitate (direkt zitierbar)
- Interviewzusammenfassung und Standpunktvergleich

【Wichtig】Die OASIS-Simulationsumgebung muss laufen, um diese Funktion nutzen zu können!"""

# ── Gliederungsplanung-Prompt ──

PLAN_SYSTEM_PROMPT = """\
Du bist ein Experte für die Erstellung von „Zukunftsprognoseberichten" mit einer „Gottesperspektive" auf die simulierte Welt – du kannst das Verhalten, die Aussagen und Interaktionen jedes Agents in der Simulation durchschauen.

【Kernkonzept】
Wir haben eine simulierte Welt aufgebaut und spezifische „Simulationsanforderungen" als Variablen eingespeist. Die Evolutionsergebnisse der simulierten Welt sind Vorhersagen darüber, was in der Zukunft passieren könnte. Was du beobachtest, sind keine „Versuchsdaten", sondern eine „Generalprobe der Zukunft".

【Deine Aufgabe】
Verfasse einen „Zukunftsprognosebericht", der folgende Fragen beantwortet:
1. Was ist unter den von uns festgelegten Bedingungen in der Zukunft passiert?
2. Wie haben die verschiedenen Agent-Gruppen (Bevölkerungsgruppen) reagiert und gehandelt?
3. Welche bemerkenswerten Zukunftstrends und Risiken hat diese Simulation aufgedeckt?

【Berichtspositionierung】
- ✅ Dies ist ein auf Simulation basierender Zukunftsprognosebericht, der aufzeigt „wenn dies geschieht, was passiert in der Zukunft"
- ✅ Fokus auf Prognoseergebnisse: Ereignisverlauf, Gruppenreaktionen, emergente Phänomene, potenzielle Risiken
- ✅ Die Aussagen und Handlungen der Agents in der simulierten Welt sind Vorhersagen über zukünftiges Gruppenverhalten
- ❌ Keine Analyse des aktuellen Zustands der realen Welt
- ❌ Keine allgemeine Zusammenfassung der öffentlichen Meinung

【Kapitelanzahl-Begrenzung】
- Mindestens 2 Kapitel, maximal 5 Kapitel
- Keine Unterkapitel erforderlich, jedes Kapitel wird direkt mit vollständigem Inhalt verfasst
- Der Inhalt soll prägnant sein und sich auf die wichtigsten Prognoseergebnisse konzentrieren
- Die Kapitelstruktur wird von dir basierend auf den Prognoseergebnissen eigenständig gestaltet

Bitte gib die Berichtsgliederung im JSON-Format aus, wie folgt:
{{
    "title": "Berichtstitel",
    "summary": "Berichtszusammenfassung (ein Satz, der die wichtigsten Prognoseergebnisse zusammenfasst)",
    "sections": [
        {{
            "title": "Kapitelüberschrift",
            "description": "Beschreibung des Kapitelinhalts"
        }}
    ]
}}

Hinweis: Das sections-Array muss mindestens 2 und maximal 5 Elemente enthalten!"""

PLAN_USER_PROMPT_TEMPLATE = """\
【Prognoseszenario-Definition】
Die in die simulierte Welt eingespeiste Variable (Simulationsanforderung): {simulation_requirement}

【Simulationswelt-Umfang】
- Anzahl der an der Simulation beteiligten Entitäten: {total_nodes}
- Anzahl der zwischen Entitäten erzeugten Beziehungen: {total_edges}
- Verteilung der Entitätstypen: {entity_types}
- Anzahl aktiver Agents: {total_entities}

【Stichprobe der von der Simulation vorhergesagten zukünftigen Fakten】
{related_facts_json}

Bitte betrachte diese Generalprobe der Zukunft aus der „Gottesperspektive":
1. Welchen Zustand zeigt die Zukunft unter den von uns festgelegten Bedingungen?
2. Wie haben die verschiedenen Bevölkerungsgruppen (Agents) reagiert und gehandelt?
3. Welche bemerkenswerten Zukunftstrends hat diese Simulation aufgedeckt?

Gestalte basierend auf den Prognoseergebnissen die am besten geeignete Berichtskapitelstruktur.

【Nochmals zur Erinnerung】Anzahl der Berichtskapitel: mindestens 2, maximal 5, der Inhalt soll prägnant sein und sich auf die wichtigsten Prognoseergebnisse konzentrieren."""

# ── Kapitelgenerierung-Prompt ──

SECTION_SYSTEM_PROMPT_TEMPLATE = """\
Du bist ein Experte für die Erstellung von „Zukunftsprognoseberichten" und verfasst gerade ein Kapitel des Berichts.

Berichtstitel: {report_title}
Berichtszusammenfassung: {report_summary}
Prognoseszenario (Simulationsanforderung): {simulation_requirement}

Aktuell zu verfassendes Kapitel: {section_title}

═══════════════════════════════════════════════════════════════
【Kernkonzept】
═══════════════════════════════════════════════════════════════

Die simulierte Welt ist eine Generalprobe der Zukunft. Wir haben spezifische Bedingungen (Simulationsanforderungen) in die simulierte Welt eingespeist.
Das Verhalten und die Interaktionen der Agents in der Simulation sind Vorhersagen über zukünftiges Gruppenverhalten.

Deine Aufgabe ist:
- Aufzuzeigen, was unter den festgelegten Bedingungen in der Zukunft passiert ist
- Vorherzusagen, wie verschiedene Bevölkerungsgruppen (Agents) reagiert und gehandelt haben
- Bemerkenswerte Zukunftstrends, Risiken und Chancen zu entdecken

❌ Schreibe keine Analyse des aktuellen Zustands der realen Welt
✅ Konzentriere dich auf „wie wird die Zukunft sein" – die Simulationsergebnisse sind die vorhergesagte Zukunft

═══════════════════════════════════════════════════════════════
【Wichtigste Regeln – Müssen eingehalten werden】
═══════════════════════════════════════════════════════════════

1. 【Werkzeuge müssen aufgerufen werden, um die simulierte Welt zu beobachten】
   - Du beobachtest die Generalprobe der Zukunft aus der „Gottesperspektive"
   - Alle Inhalte müssen aus Ereignissen und Agent-Aussagen/Handlungen der simulierten Welt stammen
   - Es ist verboten, eigenes Wissen für den Berichtsinhalt zu verwenden
   - Jedes Kapitel muss mindestens 3 Mal Werkzeuge aufrufen (maximal 5 Mal), um die simulierte Welt zu beobachten, die die Zukunft repräsentiert

2. 【Originalaussagen und -handlungen der Agents müssen zitiert werden】
   - Aussagen und Handlungen der Agents sind Vorhersagen über zukünftiges Gruppenverhalten
   - Verwende im Bericht Zitatformat, um diese Vorhersagen darzustellen, z.B.:
     > "Eine bestimmte Bevölkerungsgruppe würde sagen: Originalinhalt..."
   - Diese Zitate sind die Kernbelege der Simulationsvorhersage

3. 【Sprachkonsistenz – Zitierte Inhalte müssen in die Berichtssprache übersetzt werden】
   - Die von Werkzeugen zurückgegebenen Inhalte können in einer anderen Sprache als der Berichtssprache sein
   - Der Bericht muss vollständig in der vom Benutzer angegebenen Sprache verfasst werden
   - Wenn du Inhalte in anderen Sprachen aus Werkzeugrückgaben zitierst, musst du sie vor dem Einfügen in die Berichtssprache übersetzen
   - Beim Übersetzen die Originalbedeutung beibehalten und natürlichen, flüssigen Ausdruck sicherstellen
   - Diese Regel gilt sowohl für Fließtext als auch für Inhalte in Zitatblöcken (> Format)

4. 【Prognoseergebnisse wahrheitsgetreu darstellen】
   - Der Berichtsinhalt muss die Simulationsergebnisse der simulierten Welt widerspiegeln, die die Zukunft repräsentieren
   - Keine Informationen hinzufügen, die in der Simulation nicht existieren
   - Wenn Informationen zu einem bestimmten Aspekt unzureichend sind, dies ehrlich angeben

═══════════════════════════════════════════════════════════════
【⚠️ Formatvorgaben – Äußerst wichtig!】
═══════════════════════════════════════════════════════════════

【Ein Kapitel = Kleinste Inhaltseinheit】
- Jedes Kapitel ist die kleinste Gliederungseinheit des Berichts
- ❌ Verboten: Jegliche Markdown-Überschriften innerhalb eines Kapitels (#, ##, ###, #### usw.)
- ❌ Verboten: Kapitelhauptüberschrift am Anfang des Inhalts
- ✅ Die Kapitelüberschrift wird vom System automatisch hinzugefügt, du schreibst nur den reinen Fließtext
- ✅ Verwende **Fettdruck**, Absatztrennung, Zitate, Listen zur Inhaltsorganisation, aber keine Überschriften

【Korrektes Beispiel】
```
Dieses Kapitel analysiert die Verbreitung der öffentlichen Meinung zum Ereignis. Durch eingehende Analyse der Simulationsdaten haben wir festgestellt...

**Erste Ausbruchsphase**

Weibo übernahm als erster Schauplatz der öffentlichen Meinung die Kernfunktion der Erstveröffentlichung:

> "Weibo trug 68% des Erstveröffentlichungsvolumens bei..."

**Emotionsverstärkungsphase**

Die Douyin-Plattform verstärkte die Wirkung des Ereignisses weiter:

- Starke visuelle Wirkung
- Hohe emotionale Resonanz
```

【Falsches Beispiel】
```
## Zusammenfassung          ← Falsch! Keine Überschriften hinzufügen
### I. Erste Phase           ← Falsch! Keine ### für Unterabschnitte verwenden
#### 1.1 Detailanalyse      ← Falsch! Keine #### für Unterteilungen verwenden

Dieses Kapitel analysiert...
```

═══════════════════════════════════════════════════════════════
【Verfügbare Suchwerkzeuge】(3-5 Aufrufe pro Kapitel)
═══════════════════════════════════════════════════════════════

{tools_description}

【Empfehlungen zur Werkzeugnutzung – Bitte verschiedene Werkzeuge mischen, nicht nur eines verwenden】
- insight_forge: Tiefgehende Erkenntnisanalyse, automatische Problemzerlegung und mehrdimensionale Suche nach Fakten und Beziehungen
- panorama_search: Weitwinkel-Panoramasuche, Gesamtübersicht, Zeitverlauf und Entwicklungsprozess eines Ereignisses verstehen
- quick_search: Schnelle Überprüfung eines bestimmten Informationspunkts
- interview_agents: Simulations-Agents interviewen, Erstpersonperspektiven und echte Reaktionen verschiedener Rollen erhalten

═══════════════════════════════════════════════════════════════
【Arbeitsablauf】
═══════════════════════════════════════════════════════════════

Bei jeder Antwort kannst du nur eine der folgenden zwei Dinge tun (nicht gleichzeitig):

Option A – Werkzeug aufrufen:
Gib deine Überlegung aus, dann rufe ein Werkzeug im folgenden Format auf:
<tool_call>
{{"name": "Werkzeugname", "parameters": {{"Parametername": "Parameterwert"}}}}
</tool_call>
Das System führt das Werkzeug aus und gibt dir das Ergebnis zurück. Du musst und kannst das Werkzeugergebnis nicht selbst schreiben.

Option B – Endgültigen Inhalt ausgeben:
Wenn du genügend Informationen durch Werkzeuge gesammelt hast, gib den Kapitelinhalt mit "Final Answer:" am Anfang aus.

⚠️ Streng verboten:
- Verboten: In einer Antwort gleichzeitig Werkzeugaufruf und Final Answer
- Verboten: Werkzeugrückgaben (Observations) selbst erfinden, alle Werkzeugergebnisse werden vom System injiziert
- Maximal ein Werkzeug pro Antwort aufrufen

═══════════════════════════════════════════════════════════════
【Anforderungen an den Kapitelinhalt】
═══════════════════════════════════════════════════════════════

1. Inhalt muss auf durch Werkzeuge abgerufenen Simulationsdaten basieren
2. Reichlich Originaltext zitieren, um die Simulationsergebnisse zu zeigen
3. Markdown-Format verwenden (aber Überschriften sind verboten):
   - **Fetttext** für Hervorhebungen verwenden (anstelle von Unterüberschriften)
   - Listen (- oder 1.2.3.) zur Organisation von Kernpunkten verwenden
   - Leerzeilen zur Trennung verschiedener Absätze verwenden
   - ❌ Verboten: Jegliche Überschriftensyntax wie #, ##, ###, ####
4. 【Zitatformat-Vorgaben – Muss eigenständiger Absatz sein】
   Zitate müssen eigenständige Absätze sein, mit jeweils einer Leerzeile davor und danach, nicht in Absätze eingebettet:

   ✅ Korrektes Format:
   ```
   Die Reaktion der Schulleitung wurde als inhaltsleer angesehen.

   > "Das Reaktionsmuster der Schulleitung wirkt in der schnelllebigen Social-Media-Umgebung starr und träge."

   Diese Bewertung spiegelt die allgemeine Unzufriedenheit der Öffentlichkeit wider.
   ```

   ❌ Falsches Format:
   ```
   Die Reaktion der Schulleitung wurde als inhaltsleer angesehen. > "Das Reaktionsmuster der Schulleitung..." Diese Bewertung spiegelt...
   ```
5. Logische Kohärenz mit anderen Kapiteln bewahren
6. 【Wiederholungen vermeiden】Die unten stehenden bereits abgeschlossenen Kapitel sorgfältig lesen und keine identischen Informationen wiederholen
7. 【Nochmals betont】Keine Überschriften hinzufügen! **Fettdruck** anstelle von Unterabschnittüberschriften verwenden"""

SECTION_USER_PROMPT_TEMPLATE = """\
Bereits abgeschlossene Kapitelinhalte (bitte sorgfältig lesen, um Wiederholungen zu vermeiden):
{previous_content}

═══════════════════════════════════════════════════════════════
【Aktuelle Aufgabe】Kapitel verfassen: {section_title}
═══════════════════════════════════════════════════════════════

【Wichtige Hinweise】
1. Die oben stehenden abgeschlossenen Kapitel sorgfältig lesen, um Wiederholungen zu vermeiden!
2. Vor dem Start müssen Werkzeuge aufgerufen werden, um Simulationsdaten abzurufen
3. Bitte verschiedene Werkzeuge mischen, nicht nur eines verwenden
4. Berichtsinhalte müssen aus Suchergebnissen stammen, nicht aus eigenem Wissen

【⚠️ Format-Warnung – Muss eingehalten werden】
- ❌ Keine Überschriften schreiben (#, ##, ###, #### sind alle verboten)
- ❌ Nicht "{section_title}" als Anfang schreiben
- ✅ Kapitelüberschrift wird vom System automatisch hinzugefügt
- ✅ Direkt Fließtext schreiben, **Fettdruck** anstelle von Unterabschnittüberschriften verwenden

Bitte beginne:
1. Zuerst überlegen (Thought), welche Informationen dieses Kapitel benötigt
2. Dann Werkzeuge aufrufen (Action), um Simulationsdaten abzurufen
3. Nach dem Sammeln ausreichender Informationen Final Answer ausgeben (reiner Fließtext, keine Überschriften)"""

# ── ReACT-Schleifen-Nachrichtenvorlagen ──

REACT_OBSERVATION_TEMPLATE = """\
Observation (Suchergebnisse):

═══ Werkzeug {tool_name} Rückgabe ═══
{result}

═══════════════════════════════════════════════════════════════
Werkzeuge {tool_calls_count}/{max_tool_calls} Mal aufgerufen (verwendet: {used_tools_str}){unused_hint}
- Wenn die Informationen ausreichend sind: Kapitelinhalt mit "Final Answer:" am Anfang ausgeben (obige Originaltexte müssen zitiert werden)
- Wenn mehr Informationen benötigt werden: Ein Werkzeug aufrufen, um die Suche fortzusetzen
═══════════════════════════════════════════════════════════════"""

REACT_INSUFFICIENT_TOOLS_MSG = (
    "【Hinweis】Du hast nur {tool_calls_count} Mal Werkzeuge aufgerufen, mindestens {min_tool_calls} Mal erforderlich."
    "Bitte rufe weitere Werkzeuge auf, um mehr Simulationsdaten zu erhalten, bevor du Final Answer ausgibst. {unused_hint}"
)

REACT_INSUFFICIENT_TOOLS_MSG_ALT = (
    "Aktuell wurden nur {tool_calls_count} Mal Werkzeuge aufgerufen, mindestens {min_tool_calls} Mal erforderlich."
    "Bitte rufe Werkzeuge auf, um Simulationsdaten abzurufen. {unused_hint}"
)

REACT_TOOL_LIMIT_MSG = (
    "Die maximale Anzahl an Werkzeugaufrufen wurde erreicht ({tool_calls_count}/{max_tool_calls}), es können keine weiteren Werkzeuge aufgerufen werden."
    'Bitte gib sofort basierend auf den bereits erhaltenen Informationen den Kapitelinhalt mit "Final Answer:" am Anfang aus.'
)

REACT_UNUSED_TOOLS_HINT = "\n💡 Du hast noch nicht verwendet: {unused_list}. Es wird empfohlen, verschiedene Werkzeuge auszuprobieren, um Informationen aus mehreren Perspektiven zu erhalten"

REACT_FORCE_FINAL_MSG = "Das Limit für Werkzeugaufrufe wurde erreicht. Bitte gib direkt Final Answer: aus und generiere den Kapitelinhalt."

# ── Chat-Prompt ──

CHAT_SYSTEM_PROMPT_TEMPLATE = """\
Du bist ein prägnanter und effizienter Simulationsprognose-Assistent.

【Hintergrund】
Prognosebedingungen: {simulation_requirement}

【Bereits erstellter Analysebericht】
{report_content}

【Regeln】
1. Fragen bevorzugt basierend auf dem obigen Berichtsinhalt beantworten
2. Fragen direkt beantworten, ausschweifende Überlegungen vermeiden
3. Nur wenn der Berichtsinhalt zur Beantwortung nicht ausreicht, Werkzeuge für zusätzliche Datensuche aufrufen
4. Antworten sollen prägnant, klar und gut strukturiert sein

【Verfügbare Werkzeuge】(nur bei Bedarf verwenden, maximal 1-2 Mal aufrufen)
{tools_description}

【Werkzeugaufruf-Format】
<tool_call>
{{"name": "Werkzeugname", "parameters": {{"Parametername": "Parameterwert"}}}}
</tool_call>

【Antwortstil】
- Prägnant und direkt, keine langen Abhandlungen
- > Format für Schlüsselzitate verwenden
- Zuerst Schlussfolgerung, dann Begründung"""

CHAT_OBSERVATION_SUFFIX = "\n\nBitte beantworte die Frage kurz und prägnant."


# ═══════════════════════════════════════════════════════════════
# ReportAgent-Hauptklasse
# ═══════════════════════════════════════════════════════════════


class ReportAgent:
    """
    Report Agent – Simulations-Berichtserstellungs-Agent

    Verwendet den ReACT-Modus (Reasoning + Acting):
    1. Planungsphase: Simulationsanforderungen analysieren und Berichtsgliederung planen
    2. Generierungsphase: Kapitelweise Inhalt generieren, jedes Kapitel kann mehrfach Werkzeuge aufrufen
    3. Reflexionsphase: Vollständigkeit und Genauigkeit des Inhalts überprüfen
    """
    
    # Maximale Werkzeugaufrufe (pro Kapitel)
    MAX_TOOL_CALLS_PER_SECTION = 5
    
    # Maximale Reflexionsrunden
    MAX_REFLECTION_ROUNDS = 3
    
    # Maximale Werkzeugaufrufe im Dialog
    MAX_TOOL_CALLS_PER_CHAT = 2
    
    def __init__(
        self, 
        graph_id: str,
        simulation_id: str,
        simulation_requirement: str,
        llm_client: Optional[LLMClient] = None,
        zep_tools: Optional[ZepToolsService] = None
    ):
        """
        Report Agent initialisieren
        
        Args:
            graph_id: Graph-ID
            simulation_id: Simulations-ID
            simulation_requirement: Beschreibung der Simulationsanforderung
            llm_client: LLM-Client (optional)
            zep_tools: Zep-Werkzeugdienst (optional)
        """
        self.graph_id = graph_id
        self.simulation_id = simulation_id
        self.simulation_requirement = simulation_requirement
        
        self.llm = llm_client or LLMClient()
        self.zep_tools = zep_tools or ZepToolsService()
        
        # Werkzeugdefinitionen
        self.tools = self._define_tools()
        
        # Protokollierer (wird in generate_report initialisiert)
        self.report_logger: Optional[ReportLogger] = None
        # Konsolen-Protokollierer (wird in generate_report initialisiert)
        self.console_logger: Optional[ReportConsoleLogger] = None
        
        logger.info(t('report.agentInitDone', graphId=graph_id, simulationId=simulation_id))
    
    def _define_tools(self) -> Dict[str, Dict[str, Any]]:
        """Verfügbare Werkzeuge definieren"""
        return {
            "insight_forge": {
                "name": "insight_forge",
                "description": TOOL_DESC_INSIGHT_FORGE,
                "parameters": {
                    "query": "Die Frage oder das Thema, das du tiefgehend analysieren möchtest",
                    "report_context": "Kontext des aktuellen Berichtskapitels (optional, hilft bei der Generierung präziserer Teilfragen)"
                }
            },
            "panorama_search": {
                "name": "panorama_search",
                "description": TOOL_DESC_PANORAMA_SEARCH,
                "parameters": {
                    "query": "Suchabfrage, zur Relevanzreihenfolge",
                    "include_expired": "Ob veraltete/historische Inhalte einbezogen werden sollen (Standard: True)"
                }
            },
            "quick_search": {
                "name": "quick_search",
                "description": TOOL_DESC_QUICK_SEARCH,
                "parameters": {
                    "query": "Suchabfrage-Zeichenkette",
                    "limit": "Anzahl der zurückgegebenen Ergebnisse (optional, Standard: 10)"
                }
            },
            "interview_agents": {
                "name": "interview_agents",
                "description": TOOL_DESC_INTERVIEW_AGENTS,
                "parameters": {
                    "interview_topic": "Interviewthema oder Anforderungsbeschreibung (z.B.: 'Meinungen der Studenten zum Formaldehyd-Vorfall im Wohnheim erfahren')",
                    "max_agents": "Maximale Anzahl der zu interviewenden Agents (optional, Standard: 5, Maximum: 10)"
                }
            }
        }
    
    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any], report_context: str = "") -> str:
        """
        Werkzeugaufruf ausführen
        
        Args:
            tool_name: Werkzeugname
            parameters: Werkzeugparameter
            report_context: Berichtskontext (für InsightForge)
            
        Returns:
            Werkzeugausführungsergebnis (Textformat)
        """
        logger.info(t('report.executingTool', toolName=tool_name, params=parameters))
        
        try:
            if tool_name == "insight_forge":
                query = parameters.get("query", "")
                ctx = parameters.get("report_context", "") or report_context
                result = self.zep_tools.insight_forge(
                    graph_id=self.graph_id,
                    query=query,
                    simulation_requirement=self.simulation_requirement,
                    report_context=ctx
                )
                return result.to_text()
            
            elif tool_name == "panorama_search":
                # Breitbandsuche – Gesamtübersicht erhalten
                query = parameters.get("query", "")
                include_expired = parameters.get("include_expired", True)
                if isinstance(include_expired, str):
                    include_expired = include_expired.lower() in ['true', '1', 'yes']
                result = self.zep_tools.panorama_search(
                    graph_id=self.graph_id,
                    query=query,
                    include_expired=include_expired
                )
                return result.to_text()
            
            elif tool_name == "quick_search":
                # Einfache Suche – Schnellsuche
                query = parameters.get("query", "")
                limit = parameters.get("limit", 10)
                if isinstance(limit, str):
                    limit = int(limit)
                result = self.zep_tools.quick_search(
                    graph_id=self.graph_id,
                    query=query,
                    limit=limit
                )
                return result.to_text()
            
            elif tool_name == "interview_agents":
                # Tiefeninterview – Echte OASIS-Interview-API aufrufen für Simulations-Agent-Antworten (Dual-Plattform)
                interview_topic = parameters.get("interview_topic", parameters.get("query", ""))
                max_agents = parameters.get("max_agents", 5)
                if isinstance(max_agents, str):
                    max_agents = int(max_agents)
                max_agents = min(max_agents, 10)
                result = self.zep_tools.interview_agents(
                    simulation_id=self.simulation_id,
                    interview_requirement=interview_topic,
                    simulation_requirement=self.simulation_requirement,
                    max_agents=max_agents
                )
                return result.to_text()
            
            # ========== Abwärtskompatible alte Werkzeuge (interne Weiterleitung zu neuen Werkzeugen) ==========
            
            elif tool_name == "search_graph":
                # Weiterleitung zu quick_search
                logger.info(t('report.redirectToQuickSearch'))
                return self._execute_tool("quick_search", parameters, report_context)
            
            elif tool_name == "get_graph_statistics":
                result = self.zep_tools.get_graph_statistics(self.graph_id)
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_entity_summary":
                entity_name = parameters.get("entity_name", "")
                result = self.zep_tools.get_entity_summary(
                    graph_id=self.graph_id,
                    entity_name=entity_name
                )
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            elif tool_name == "get_simulation_context":
                # Weiterleitung zu insight_forge, da es leistungsfähiger ist
                logger.info(t('report.redirectToInsightForge'))
                query = parameters.get("query", self.simulation_requirement)
                return self._execute_tool("insight_forge", {"query": query}, report_context)
            
            elif tool_name == "get_entities_by_type":
                entity_type = parameters.get("entity_type", "")
                nodes = self.zep_tools.get_entities_by_type(
                    graph_id=self.graph_id,
                    entity_type=entity_type
                )
                result = [n.to_dict() for n in nodes]
                return json.dumps(result, ensure_ascii=False, indent=2)
            
            else:
                return f"Unbekanntes Werkzeug: {tool_name}. Bitte verwende eines der folgenden Werkzeuge: insight_forge, panorama_search, quick_search"
                
        except Exception as e:
            logger.error(t('report.toolExecFailed', toolName=tool_name, error=str(e)))
            return f"Werkzeugausführung fehlgeschlagen: {str(e)}"
    
    # Gültige Werkzeugnamen-Menge, zur Validierung beim Fallback-Parsing von rohem JSON
    VALID_TOOL_NAMES = {"insight_forge", "panorama_search", "quick_search", "interview_agents"}

    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """
        Werkzeugaufrufe aus der LLM-Antwort parsen

        Unterstützte Formate (nach Priorität):
        1. <tool_call>{"name": "tool_name", "parameters": {...}}</tool_call>
        2. Rohes JSON (Antwort insgesamt oder einzelne Zeile ist ein Werkzeugaufruf-JSON)
        """
        tool_calls = []

        # Format 1: XML-Stil (Standardformat)
        xml_pattern = r'<tool_call>\s*(\{.*?\})\s*</tool_call>'
        for match in re.finditer(xml_pattern, response, re.DOTALL):
            try:
                call_data = json.loads(match.group(1))
                tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        if tool_calls:
            return tool_calls

        # Format 2: Fallback – LLM gibt rohes JSON aus (ohne <tool_call>-Tags)
        # Wird nur versucht, wenn Format 1 nicht gematcht hat, um Fehlzuordnungen im Fließtext zu vermeiden
        stripped = response.strip()
        if stripped.startswith('{') and stripped.endswith('}'):
            try:
                call_data = json.loads(stripped)
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
                    return tool_calls
            except json.JSONDecodeError:
                pass

        # Antwort kann Denktext + rohes JSON enthalten, versuche das letzte JSON-Objekt zu extrahieren
        json_pattern = r'(\{"(?:name|tool)"\s*:.*?\})\s*$'
        match = re.search(json_pattern, stripped, re.DOTALL)
        if match:
            try:
                call_data = json.loads(match.group(1))
                if self._is_valid_tool_call(call_data):
                    tool_calls.append(call_data)
            except json.JSONDecodeError:
                pass

        return tool_calls

    def _is_valid_tool_call(self, data: dict) -> bool:
        """Prüfen, ob das geparste JSON ein gültiger Werkzeugaufruf ist"""
        # Unterstützt beide Schlüsselnamen: {"name": ..., "parameters": ...} und {"tool": ..., "params": ...}
        tool_name = data.get("name") or data.get("tool")
        if tool_name and tool_name in self.VALID_TOOL_NAMES:
            # Schlüsselnamen auf name / parameters vereinheitlichen
            if "tool" in data:
                data["name"] = data.pop("tool")
            if "params" in data and "parameters" not in data:
                data["parameters"] = data.pop("params")
            return True
        return False
    
    def _get_tools_description(self) -> str:
        """Werkzeugbeschreibungstext generieren"""
        desc_parts = ["Verfügbare Werkzeuge:"]
        for name, tool in self.tools.items():
            params_desc = ", ".join([f"{k}: {v}" for k, v in tool["parameters"].items()])
            desc_parts.append(f"- {name}: {tool['description']}")
            if params_desc:
                desc_parts.append(f"  Parameter: {params_desc}")
        return "\n".join(desc_parts)
    
    def plan_outline(
        self, 
        progress_callback: Optional[Callable] = None
    ) -> ReportOutline:
        """
        Berichtsgliederung planen
        
        Simulationsanforderungen mit LLM analysieren und Berichtsgliederung planen
        
        Args:
            progress_callback: Fortschritts-Callback-Funktion
            
        Returns:
            ReportOutline: Berichtsgliederung
        """
        logger.info(t('report.startPlanningOutline'))
        
        if progress_callback:
            progress_callback("planning", 0, t('progress.analyzingRequirements'))
        
        # Zunächst Simulationskontext abrufen
        context = self.zep_tools.get_simulation_context(
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement
        )
        
        if progress_callback:
            progress_callback("planning", 30, t('progress.generatingOutline'))
        
        system_prompt = f"{PLAN_SYSTEM_PROMPT}\n\n{get_language_instruction()}"
        user_prompt = PLAN_USER_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            total_nodes=context.get('graph_statistics', {}).get('total_nodes', 0),
            total_edges=context.get('graph_statistics', {}).get('total_edges', 0),
            entity_types=list(context.get('graph_statistics', {}).get('entity_types', {}).keys()),
            total_entities=context.get('total_entities', 0),
            related_facts_json=json.dumps(context.get('related_facts', [])[:10], ensure_ascii=False, indent=2),
        )

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            if progress_callback:
                progress_callback("planning", 80, t('progress.parsingOutline'))
            
            # Gliederung parsen
            sections = []
            for section_data in response.get("sections", []):
                sections.append(ReportSection(
                    title=section_data.get("title", ""),
                    content=""
                ))
            
            outline = ReportOutline(
                title=response.get("title", "Simulationsanalysebericht"),
                summary=response.get("summary", ""),
                sections=sections
            )
            
            if progress_callback:
                progress_callback("planning", 100, t('progress.outlinePlanComplete'))
            
            logger.info(t('report.outlinePlanDone', count=len(sections)))
            return outline
            
        except Exception as e:
            logger.error(t('report.outlinePlanFailed', error=str(e)))
            # Standardgliederung zurückgeben (3 Kapitel, als Fallback)
            return ReportOutline(
                title="Zukunftsprognosebericht",
                summary="Analyse zukünftiger Trends und Risiken basierend auf Simulationsprognosen",
                sections=[
                    ReportSection(title="Prognoseszenario und Kernerkenntnisse"),
                    ReportSection(title="Analyse der vorhergesagten Gruppenverhalten"),
                    ReportSection(title="Trendausblick und Risikohinweise")
                ]
            )
    
    def _generate_section_react(
        self, 
        section: ReportSection,
        outline: ReportOutline,
        previous_sections: List[str],
        progress_callback: Optional[Callable] = None,
        section_index: int = 0
    ) -> str:
        """
        Einzelnes Kapitel im ReACT-Modus generieren
        
        ReACT-Schleife:
        1. Thought (Denken) – Analysieren, welche Informationen benötigt werden
        2. Action (Handeln) – Werkzeuge aufrufen, um Informationen abzurufen
        3. Observation (Beobachten) – Werkzeugergebnisse analysieren
        4. Wiederholen bis genügend Informationen vorhanden oder Maximum erreicht
        5. Final Answer (Endantwort) – Kapitelinhalt generieren
        
        Args:
            section: Zu generierendes Kapitel
            outline: Vollständige Gliederung
            previous_sections: Inhalte vorheriger Kapitel (für Kohärenz)
            progress_callback: Fortschritts-Callback
            section_index: Kapitelindex (für Protokollierung)
            
        Returns:
            Kapitelinhalt (Markdown-Format)
        """
        logger.info(t('report.reactGenerateSection', title=section.title))
        
        # Kapitelstart protokollieren
        if self.report_logger:
            self.report_logger.log_section_start(section.title, section_index)
        
        system_prompt = SECTION_SYSTEM_PROMPT_TEMPLATE.format(
            report_title=outline.title,
            report_summary=outline.summary,
            simulation_requirement=self.simulation_requirement,
            section_title=section.title,
            tools_description=self._get_tools_description(),
        )
        system_prompt = f"{system_prompt}\n\n{get_language_instruction()}"

        # Benutzer-Prompt erstellen – jedes abgeschlossene Kapitel max. 4000 Zeichen
        if previous_sections:
            previous_parts = []
            for sec in previous_sections:
                # Jedes Kapitel maximal 4000 Zeichen
                truncated = sec[:4000] + "..." if len(sec) > 4000 else sec
                previous_parts.append(truncated)
            previous_content = "\n\n---\n\n".join(previous_parts)
        else:
            previous_content = "(Dies ist das erste Kapitel)"
        
        user_prompt = SECTION_USER_PROMPT_TEMPLATE.format(
            previous_content=previous_content,
            section_title=section.title,
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        # ReACT-Schleife
        tool_calls_count = 0
        max_iterations = 5  # Maximale Iterationsrunden
        min_tool_calls = 3  # Mindestanzahl Werkzeugaufrufe
        conflict_retries = 0  # Aufeinanderfolgende Konfliktzählung bei gleichzeitigem Werkzeugaufruf und Final Answer
        used_tools = set()  # Aufgezeichnete bereits aufgerufene Werkzeugnamen
        all_tools = {"insight_forge", "panorama_search", "quick_search", "interview_agents"}

        # Berichtskontext für InsightForge-Teilfragen-Generierung
        report_context = f"Kapitelüberschrift: {section.title}\nSimulationsanforderung: {self.simulation_requirement}"
        
        for iteration in range(max_iterations):
            if progress_callback:
                progress_callback(
                    "generating", 
                    int((iteration / max_iterations) * 100),
                    t('progress.deepSearchAndWrite', current=tool_calls_count, max=self.MAX_TOOL_CALLS_PER_SECTION)
                )
            
            # LLM aufrufen
            response = self.llm.chat(
                messages=messages,
                temperature=0.5,
                max_tokens=4096
            )

            # Prüfen, ob LLM-Rückgabe None ist (API-Fehler oder leerer Inhalt)
            if response is None:
                logger.warning(t('report.sectionIterNone', title=section.title, iteration=iteration + 1))
                # Wenn noch Iterationen übrig, Nachricht hinzufügen und erneut versuchen
                if iteration < max_iterations - 1:
                    messages.append({"role": "assistant", "content": "(Antwort war leer)"})
                    messages.append({"role": "user", "content": "Bitte fahre mit der Inhaltsgenerierung fort."})
                    continue
                # Letzte Iteration ebenfalls None, Schleife verlassen für erzwungenen Abschluss
                break

            logger.debug(f"LLM-Antwort: {response[:200]}...")

            # Einmal parsen, Ergebnis wiederverwenden
            tool_calls = self._parse_tool_calls(response)
            has_tool_calls = bool(tool_calls)
            has_final_answer = "Final Answer:" in response

            # ── Konfliktbehandlung: LLM hat gleichzeitig Werkzeugaufruf und Final Answer ausgegeben ──
            if has_tool_calls and has_final_answer:
                conflict_retries += 1
                logger.warning(
                    t('report.sectionConflict', title=section.title, iteration=iteration+1, conflictCount=conflict_retries)
                )

                if conflict_retries <= 2:
                    # Erste zwei Male: Diese Antwort verwerfen, LLM zur erneuten Antwort auffordern
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": (
                            "【Formatfehler】Du hast in einer Antwort gleichzeitig einen Werkzeugaufruf und Final Answer eingefügt, das ist nicht erlaubt.\n"
                            "Jede Antwort darf nur eine der folgenden zwei Dinge tun:\n"
                            "- Ein Werkzeug aufrufen (einen <tool_call>-Block ausgeben, kein Final Answer schreiben)\n"
                            "- Endgültigen Inhalt ausgeben (mit 'Final Answer:' beginnen, kein <tool_call> einschließen)\n"
                            "Bitte antworte erneut und tue nur eines davon."
                        ),
                    })
                    continue
                else:
                    # Drittes Mal: Degradierte Behandlung, zum ersten Werkzeugaufruf abschneiden und erzwungen ausführen
                    logger.warning(
                        t('report.sectionConflictDowngrade', title=section.title, conflictCount=conflict_retries)
                    )
                    first_tool_end = response.find('</tool_call>')
                    if first_tool_end != -1:
                        response = response[:first_tool_end + len('</tool_call>')]
                        tool_calls = self._parse_tool_calls(response)
                        has_tool_calls = bool(tool_calls)
                    has_final_answer = False
                    conflict_retries = 0

            # LLM-Antwort protokollieren
            if self.report_logger:
                self.report_logger.log_llm_response(
                    section_title=section.title,
                    section_index=section_index,
                    response=response,
                    iteration=iteration + 1,
                    has_tool_calls=has_tool_calls,
                    has_final_answer=has_final_answer
                )

            # ── Fall 1: LLM hat Final Answer ausgegeben ──
            if has_final_answer:
                # Werkzeugaufrufe nicht ausreichend, ablehnen und weitere Werkzeugaufrufe fordern
                if tool_calls_count < min_tool_calls:
                    messages.append({"role": "assistant", "content": response})
                    unused_tools = all_tools - used_tools
                    unused_hint = f"(Diese Werkzeuge wurden noch nicht verwendet, empfohlen sie auszuprobieren: {', '.join(unused_tools)})" if unused_tools else ""
                final_answer = response.split("Final Answer:")[-1].strip()
                logger.info(t('report.sectionGenDone', title=section.title, count=tool_calls_count))

                if self.report_logger:
                    self.report_logger.log_section_content(
                        section_title=section.title,
                        section_index=section_index,
                        content=final_answer,
                        tool_calls_count=tool_calls_count
                    )
                return final_answer

            # ── Fall 2: LLM versucht Werkzeug aufzurufen ──
            if has_tool_calls:
                # Werkzeugkontingent erschöpft → explizit mitteilen, Final Answer anfordern
                if tool_calls_count >= self.MAX_TOOL_CALLS_PER_SECTION:
                    messages.append({"role": "assistant", "content": response})
                    messages.append({
                        "role": "user",
                        "content": REACT_TOOL_LIMIT_MSG.format(
                            tool_calls_count=tool_calls_count,
                            max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        ),
                    })
                    continue

                # Nur den ersten Werkzeugaufruf ausführen
                call = tool_calls[0]
                if len(tool_calls) > 1:
                    logger.info(t('report.multiToolOnlyFirst', total=len(tool_calls), toolName=call['name']))

                if self.report_logger:
                    self.report_logger.log_tool_call(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        parameters=call.get("parameters", {}),
                        iteration=iteration + 1
                    )

                result = self._execute_tool(
                    call["name"],
                    call.get("parameters", {}),
                    report_context=report_context
                )

                if self.report_logger:
                    self.report_logger.log_tool_result(
                        section_title=section.title,
                        section_index=section_index,
                        tool_name=call["name"],
                        result=result,
                        iteration=iteration + 1
                    )

                tool_calls_count += 1
                used_tools.add(call['name'])

                # Hinweis zu nicht verwendeten Werkzeugen erstellen
                unused_tools = all_tools - used_tools
                unused_hint = ""
                if unused_tools and tool_calls_count < self.MAX_TOOL_CALLS_PER_SECTION:
                    unused_hint = REACT_UNUSED_TOOLS_HINT.format(unused_list="、".join(unused_tools))

                messages.append({"role": "assistant", "content": response})
                messages.append({
                    "role": "user",
                    "content": REACT_OBSERVATION_TEMPLATE.format(
                        tool_name=call["name"],
                        result=result,
                        tool_calls_count=tool_calls_count,
                        max_tool_calls=self.MAX_TOOL_CALLS_PER_SECTION,
                        used_tools_str=", ".join(used_tools),
                        unused_hint=unused_hint,
                    ),
                })
                continue

            # ── Fall 3: Weder Werkzeugaufruf noch Final Answer ──
            messages.append({"role": "assistant", "content": response})

            if tool_calls_count < min_tool_calls:
                # Werkzeugaufrufe nicht ausreichend, nicht verwendete Werkzeuge empfehlen
                unused_tools = all_tools - used_tools
                unused_hint = f"(Diese Werkzeuge wurden noch nicht verwendet, empfohlen sie auszuprobieren: {', '.join(unused_tools)})" if unused_tools else ""

                messages.append({
                    "role": "user",
                    "content": REACT_INSUFFICIENT_TOOLS_MSG_ALT.format(
                        tool_calls_count=tool_calls_count,
                        min_tool_calls=min_tool_calls,
                        unused_hint=unused_hint,
                    ),
                })
                continue

            # Werkzeugaufrufe ausreichend, LLM hat Inhalt ohne "Final Answer:"-Präfix ausgegeben
            # Diesen Inhalt direkt als endgültige Antwort verwenden
            logger.info(t('report.sectionNoPrefix', title=section.title, count=tool_calls_count))
            final_answer = response.strip()

            if self.report_logger:
                self.report_logger.log_section_content(
                    section_title=section.title,
                    section_index=section_index,
                    content=final_answer,
                    tool_calls_count=tool_calls_count
                )
            return final_answer
        
        # Maximale Iterationsanzahl erreicht, Inhaltsgenerierung erzwingen
        logger.warning(t('report.sectionMaxIter', title=section.title))
        messages.append({"role": "user", "content": REACT_FORCE_FINAL_MSG})
        
        response = self.llm.chat(
            messages=messages,
            temperature=0.5,
            max_tokens=4096
        )

        # Prüfen, ob LLM bei erzwungenem Abschluss None zurückgibt
        if response is None:
            logger.error(t('report.sectionForceFailed', title=section.title))
            final_answer = t('report.sectionGenFailedContent')
        elif "Final Answer:" in response:
            final_answer = response.split("Final Answer:")[-1].strip()
        else:
            final_answer = response
        
        # Protokoll: Kapitelinhalt-Generierung abgeschlossen
        if self.report_logger:
            self.report_logger.log_section_content(
                section_title=section.title,
                section_index=section_index,
                content=final_answer,
                tool_calls_count=tool_calls_count
            )
        
        return final_answer
    
    def generate_report(
        self, 
        progress_callback: Optional[Callable[[str, int, str], None]] = None,
        report_id: Optional[str] = None
    ) -> Report:
        """
        Vollständigen Bericht generieren (kapitelweise Echtzeitausgabe)
        
        Jedes Kapitel wird nach Fertigstellung sofort im Ordner gespeichert, ohne auf den gesamten Bericht zu warten.
        Dateistruktur:
        reports/{report_id}/
            meta.json       - Bericht-Metainformationen
            outline.json    - Berichtsgliederung
            progress.json   - Generierungsfortschritt
            section_01.md   - Kapitel 1
            section_02.md   - Kapitel 2
            ...
            full_report.md  - Vollständiger Bericht
        
        Args:
            progress_callback: Fortschritts-Callback-Funktion (stage, progress, message)
            report_id: Berichts-ID (optional, wird automatisch generiert wenn nicht angegeben)
            
        Returns:
            Report: Vollständiger Bericht
        """
        import uuid
        
        # Wenn keine report_id übergeben wurde, automatisch generieren
        if not report_id:
            report_id = f"report_{uuid.uuid4().hex[:12]}"
        start_time = datetime.now()
        
        report = Report(
            report_id=report_id,
            simulation_id=self.simulation_id,
            graph_id=self.graph_id,
            simulation_requirement=self.simulation_requirement,
            status=ReportStatus.PENDING,
            created_at=datetime.now().isoformat()
        )
        
        # Liste der abgeschlossenen Kapitelüberschriften (für Fortschrittsverfolgung)
        completed_section_titles = []
        
        try:
            # Initialisierung: Berichtsordner erstellen und Anfangsstatus speichern
            ReportManager._ensure_report_folder(report_id)
            
            # Protokollierer initialisieren (strukturiertes Protokoll agent_log.jsonl)
            self.report_logger = ReportLogger(report_id)
            self.report_logger.log_start(
                simulation_id=self.simulation_id,
                graph_id=self.graph_id,
                simulation_requirement=self.simulation_requirement
            )
            
            # Konsolen-Protokollierer initialisieren (console_log.txt)
            self.console_logger = ReportConsoleLogger(report_id)
            
            ReportManager.update_progress(
                report_id, "pending", 0, t('progress.initReport'),
                completed_sections=[]
            )
            ReportManager.save_report(report)
            
            # Phase 1: Gliederung planen
            report.status = ReportStatus.PLANNING
            ReportManager.update_progress(
                report_id, "planning", 5, t('progress.startPlanningOutline'),
                completed_sections=[]
            )
            
            # Planungsbeginn protokollieren
            self.report_logger.log_planning_start()
            
            if progress_callback:
                progress_callback("planning", 0, t('progress.startPlanningOutline'))
            
            outline = self.plan_outline(
                progress_callback=lambda stage, prog, msg: 
                    progress_callback(stage, prog // 5, msg) if progress_callback else None
            )
            report.outline = outline
            
            # Planungsabschluss protokollieren
            self.report_logger.log_planning_complete(outline.to_dict())
            
            # Gliederung in Datei speichern
            ReportManager.save_outline(report_id, outline)
            ReportManager.update_progress(
                report_id, "planning", 15, t('progress.outlineDone', count=len(outline.sections)),
                completed_sections=[]
            )
            ReportManager.save_report(report)
            
            logger.info(t('report.outlineSavedToFile', reportId=report_id))
            
            # Phase 2: Kapitelweise Generierung (kapitelweise Speicherung)
            report.status = ReportStatus.GENERATING
            
            total_sections = len(outline.sections)
            generated_sections = []  # Inhalt für Kontext speichern
            
            for i, section in enumerate(outline.sections):
                section_num = i + 1
                base_progress = 20 + int((i / total_sections) * 70)
                
                # Fortschritt aktualisieren
                ReportManager.update_progress(
                    report_id, "generating", base_progress,
                    t('progress.generatingSection', title=section.title, current=section_num, total=total_sections),
                    current_section=section.title,
                    completed_sections=completed_section_titles
                )

                if progress_callback:
                    progress_callback(
                        "generating",
                        base_progress,
                        t('progress.generatingSection', title=section.title, current=section_num, total=total_sections)
                    )
                
                # Hauptkapitelinhalt generieren
                section_content = self._generate_section_react(
                    section=section,
                    outline=outline,
                    previous_sections=generated_sections,
                    progress_callback=lambda stage, prog, msg:
                        progress_callback(
                            stage, 
                            base_progress + int(prog * 0.7 / total_sections),
                            msg
                        ) if progress_callback else None,
                    section_index=section_num
                )
                
                section.content = section_content
                generated_sections.append(f"## {section.title}\n\n{section_content}")

                # Kapitel speichern
                ReportManager.save_section(report_id, section_num, section)
                completed_section_titles.append(section.title)

                # Kapitelabschluss protokollieren
                full_section_content = f"## {section.title}\n\n{section_content}"

                if self.report_logger:
                    self.report_logger.log_section_full_complete(
                        section_title=section.title,
                        section_index=section_num,
                        full_content=full_section_content.strip()
                    )

                logger.info(t('report.sectionSaved', reportId=report_id, sectionNum=f"{section_num:02d}"))
                
                # Fortschritt aktualisieren
                ReportManager.update_progress(
                    report_id, "generating", 
                    base_progress + int(70 / total_sections),
                    t('progress.sectionDone', title=section.title),
                    current_section=None,
                    completed_sections=completed_section_titles
                )
            
            # Phase 3: Vollständigen Bericht zusammenstellen
            if progress_callback:
                progress_callback("generating", 95, t('progress.assemblingReport'))
            
            ReportManager.update_progress(
                report_id, "generating", 95, t('progress.assemblingReport'),
                completed_sections=completed_section_titles
            )
            
            # ReportManager zum Zusammenstellen des vollständigen Berichts verwenden
            report.markdown_content = ReportManager.assemble_full_report(report_id, outline)
            report.status = ReportStatus.COMPLETED
            report.completed_at = datetime.now().isoformat()
            
            # Gesamtdauer berechnen
            total_time_seconds = (datetime.now() - start_time).total_seconds()
            
            # Berichtsabschluss protokollieren
            if self.report_logger:
                self.report_logger.log_report_complete(
                    total_sections=total_sections,
                    total_time_seconds=total_time_seconds
                )
            
            # Endbericht speichern
            ReportManager.save_report(report)
            ReportManager.update_progress(
                report_id, "completed", 100, t('progress.reportComplete'),
                completed_sections=completed_section_titles
            )
            
            if progress_callback:
                progress_callback("completed", 100, t('progress.reportComplete'))
            
            logger.info(t('report.reportGenDone', reportId=report_id))
            
            # Konsolen-Protokollierer schließen
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None
            
            return report
            
        except Exception as e:
            logger.error(t('report.reportGenFailed', error=str(e)))
            report.status = ReportStatus.FAILED
            report.error = str(e)
            
            # Fehler protokollieren
            if self.report_logger:
                self.report_logger.log_error(str(e), "failed")
            
            # Fehlerstatus speichern
            try:
                ReportManager.save_report(report)
                ReportManager.update_progress(
                    report_id, "failed", -1, t('progress.reportFailed', error=str(e)),
                    completed_sections=completed_section_titles
                )
            except Exception:
                pass  # Fehler beim Speichern ignorieren
            
            # Konsolen-Protokollierer schließen
            if self.console_logger:
                self.console_logger.close()
                self.console_logger = None
            
            return report
    
    def chat(
        self, 
        message: str,
        chat_history: List[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Mit dem Report Agent chatten
        
        Im Dialog kann der Agent autonom Suchwerkzeuge aufrufen, um Fragen zu beantworten
        
        Args:
            message: Benutzernachricht
            chat_history: Dialogverlauf
            
        Returns:
            {
                "response": "Agent-Antwort",
                "tool_calls": [Liste der aufgerufenen Werkzeuge],
                "sources": [Informationsquellen]
            }
        """
        logger.info(t('report.agentChat', message=message[:50]))
        
        chat_history = chat_history or []
        
        # Bereits generierten Berichtsinhalt abrufen
        report_content = ""
        try:
            report = ReportManager.get_report_by_simulation(self.simulation_id)
            if report and report.markdown_content:
                # Berichtslänge begrenzen, um zu langen Kontext zu vermeiden
                report_content = report.markdown_content[:15000]
                if len(report.markdown_content) > 15000:
                    report_content += "\n\n... [Berichtsinhalt wurde gekürzt] ..."
        except Exception as e:
            logger.warning(t('report.fetchReportFailed', error=e))
        
        system_prompt = CHAT_SYSTEM_PROMPT_TEMPLATE.format(
            simulation_requirement=self.simulation_requirement,
            report_content=report_content if report_content else "(Noch kein Bericht vorhanden)",
            tools_description=self._get_tools_description(),
        )
        system_prompt = f"{system_prompt}\n\n{get_language_instruction()}"

        # Nachrichten erstellen
        messages = [{"role": "system", "content": system_prompt}]
        
        # Dialogverlauf hinzufügen
        for h in chat_history[-10:]:  # Verlaufslänge begrenzen
            messages.append(h)
        
        # Benutzernachricht hinzufügen
        messages.append({
            "role": "user", 
            "content": message
        })
        
        # ReACT-Schleife (vereinfachte Version)
        tool_calls_made = []
        max_iterations = 2  # Reduzierte Iterationsrunden
        
        for iteration in range(max_iterations):
            response = self.llm.chat(
                messages=messages,
                temperature=0.5
            )
            
            # Werkzeugaufrufe parsen
            tool_calls = self._parse_tool_calls(response)
            
            if not tool_calls:
                # Keine Werkzeugaufrufe, Antwort direkt zurückgeben
                clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', response, flags=re.DOTALL)
                clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)
                
                return {
                    "response": clean_response.strip(),
                    "tool_calls": tool_calls_made,
                    "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
                }
            
            # Werkzeugaufrufe ausführen (Anzahl begrenzt)
            tool_results = []
            for call in tool_calls[:1]:  # Maximal 1 Werkzeugaufruf pro Runde
                if len(tool_calls_made) >= self.MAX_TOOL_CALLS_PER_CHAT:
                    break
                result = self._execute_tool(call["name"], call.get("parameters", {}))
                tool_results.append({
                    "tool": call["name"],
                    "result": result[:1500]  # Ergebnislänge begrenzen
                })
                tool_calls_made.append(call)
            
            # Ergebnisse zu Nachrichten hinzufügen
            messages.append({"role": "assistant", "content": response})
            observation = "\n".join([f"[{r['tool']}-Ergebnis]\n{r['result']}" for r in tool_results])
            messages.append({
                "role": "user",
                "content": observation + CHAT_OBSERVATION_SUFFIX
            })
        
        # Maximale Iteration erreicht, endgültige Antwort abrufen
        final_response = self.llm.chat(
            messages=messages,
            temperature=0.5
        )
        
        # Antwort bereinigen
        clean_response = re.sub(r'<tool_call>.*?</tool_call>', '', final_response, flags=re.DOTALL)
        clean_response = re.sub(r'\[TOOL_CALL\].*?\)', '', clean_response)
        
        return {
            "response": clean_response.strip(),
            "tool_calls": tool_calls_made,
            "sources": [tc.get("parameters", {}).get("query", "") for tc in tool_calls_made]
        }


class ReportManager:
    """
    Berichtsmanager
    
    Zuständig für die persistente Speicherung und das Abrufen von Berichten
    
    Dateistruktur (kapitelweise Ausgabe):
    reports/
      {report_id}/
        meta.json          - Bericht-Metainformationen und Status
        outline.json       - Berichtsgliederung
        progress.json      - Generierungsfortschritt
        section_01.md      - Kapitel 1
        section_02.md      - Kapitel 2
        ...
        full_report.md     - Vollständiger Bericht
    """
    
    # Berichtsspeicherverzeichnis
    REPORTS_DIR = os.path.join(Config.UPLOAD_FOLDER, 'reports')
    
    @classmethod
    def _ensure_reports_dir(cls):
        """Sicherstellen, dass das Berichtsstammverzeichnis existiert"""
        os.makedirs(cls.REPORTS_DIR, exist_ok=True)
    
    @classmethod
    def _get_report_folder(cls, report_id: str) -> str:
        """Berichtsordnerpfad abrufen"""
        return os.path.join(cls.REPORTS_DIR, report_id)
    
    @classmethod
    def _ensure_report_folder(cls, report_id: str) -> str:
        """Sicherstellen, dass der Berichtsordner existiert, und Pfad zurückgeben"""
        folder = cls._get_report_folder(report_id)
        os.makedirs(folder, exist_ok=True)
        return folder
    
    @classmethod
    def _get_report_path(cls, report_id: str) -> str:
        """Pfad der Bericht-Metainformationsdatei abrufen"""
        return os.path.join(cls._get_report_folder(report_id), "meta.json")
    
    @classmethod
    def _get_report_markdown_path(cls, report_id: str) -> str:
        """Pfad der vollständigen Bericht-Markdown-Datei abrufen"""
        return os.path.join(cls._get_report_folder(report_id), "full_report.md")
    
    @classmethod
    def _get_outline_path(cls, report_id: str) -> str:
        """Gliederungsdateipfad abrufen"""
        return os.path.join(cls._get_report_folder(report_id), "outline.json")
    
    @classmethod
    def _get_progress_path(cls, report_id: str) -> str:
        """Fortschrittsdateipfad abrufen"""
        return os.path.join(cls._get_report_folder(report_id), "progress.json")
    
    @classmethod
    def _get_section_path(cls, report_id: str, section_index: int) -> str:
        """Kapitel-Markdown-Dateipfad abrufen"""
        return os.path.join(cls._get_report_folder(report_id), f"section_{section_index:02d}.md")
    
    @classmethod
    def _get_agent_log_path(cls, report_id: str) -> str:
        """Agent-Protokolldateipfad abrufen"""
        return os.path.join(cls._get_report_folder(report_id), "agent_log.jsonl")
    
    @classmethod
    def _get_console_log_path(cls, report_id: str) -> str:
        """Konsolen-Protokolldateipfad abrufen"""
        return os.path.join(cls._get_report_folder(report_id), "console_log.txt")
    
    @classmethod
    def get_console_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Konsolen-Protokollinhalt abrufen
        
        Dies ist das Konsolen-Ausgabeprotokoll (INFO, WARNING usw.) während der Berichtserstellung,
        unterschiedlich vom strukturierten Protokoll in agent_log.jsonl.
        
        Args:
            report_id: Berichts-ID
            from_line: Ab welcher Zeile gelesen werden soll (für inkrementelles Abrufen, 0 = von Anfang an)
            
        Returns:
            {
                "logs": [Liste der Protokollzeilen],
                "total_lines": Gesamtanzahl der Zeilen,
                "from_line": Startzeile,
                "has_more": Ob weitere Protokolle vorhanden sind
            }
        """
        log_path = cls._get_console_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    # Ursprüngliche Protokollzeile beibehalten, Zeilenumbruch am Ende entfernen
                    logs.append(line.rstrip('\n\r'))
        
        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # Bis zum Ende gelesen
        }
    
    @classmethod
    def get_console_log_stream(cls, report_id: str) -> List[str]:
        """
        Vollständiges Konsolenprotokoll abrufen (alles auf einmal)
        
        Args:
            report_id: Berichts-ID
            
        Returns:
            Liste der Protokollzeilen
        """
        result = cls.get_console_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def get_agent_log(cls, report_id: str, from_line: int = 0) -> Dict[str, Any]:
        """
        Agent-Protokollinhalt abrufen
        
        Args:
            report_id: Berichts-ID
            from_line: Ab welcher Zeile gelesen werden soll (für inkrementelles Abrufen, 0 = von Anfang an)
            
        Returns:
            {
                "logs": [Liste der Protokolleinträge],
                "total_lines": Gesamtanzahl der Zeilen,
                "from_line": Startzeile,
                "has_more": Ob weitere Protokolle vorhanden sind
            }
        """
        log_path = cls._get_agent_log_path(report_id)
        
        if not os.path.exists(log_path):
            return {
                "logs": [],
                "total_lines": 0,
                "from_line": 0,
                "has_more": False
            }
        
        logs = []
        total_lines = 0
        
        with open(log_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total_lines = i + 1
                if i >= from_line:
                    try:
                        log_entry = json.loads(line.strip())
                        logs.append(log_entry)
                    except json.JSONDecodeError:
                        # Zeilen mit fehlgeschlagenem Parsing überspringen
                        continue
        
        return {
            "logs": logs,
            "total_lines": total_lines,
            "from_line": from_line,
            "has_more": False  # Bis zum Ende gelesen
        }
    
    @classmethod
    def get_agent_log_stream(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Vollständiges Agent-Protokoll abrufen (alles auf einmal)
        
        Args:
            report_id: Berichts-ID
            
        Returns:
            Liste der Protokolleinträge
        """
        result = cls.get_agent_log(report_id, from_line=0)
        return result["logs"]
    
    @classmethod
    def save_outline(cls, report_id: str, outline: ReportOutline) -> None:
        """
        Berichtsgliederung speichern
        
        Wird sofort nach Abschluss der Planungsphase aufgerufen
        """
        cls._ensure_report_folder(report_id)
        
        with open(cls._get_outline_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(outline.to_dict(), f, ensure_ascii=False, indent=2)
        
        logger.info(t('report.outlineSaved', reportId=report_id))
    
    @classmethod
    def save_section(
        cls,
        report_id: str,
        section_index: int,
        section: ReportSection
    ) -> str:
        """
        Einzelnes Kapitel speichern

        Wird sofort nach Fertigstellung jedes Kapitels aufgerufen, für kapitelweise Ausgabe

        Args:
            report_id: Berichts-ID
            section_index: Kapitelindex (ab 1)
            section: Kapitelobjekt

        Returns:
            Gespeicherter Dateipfad
        """
        cls._ensure_report_folder(report_id)

        # Kapitel-Markdown-Inhalt erstellen – mögliche doppelte Überschriften bereinigen
        cleaned_content = cls._clean_section_content(section.content, section.title)
        md_content = f"## {section.title}\n\n"
        if cleaned_content:
            md_content += f"{cleaned_content}\n\n"

        # Datei speichern
        file_suffix = f"section_{section_index:02d}.md"
        file_path = os.path.join(cls._get_report_folder(report_id), file_suffix)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)

        logger.info(t('report.sectionFileSaved', reportId=report_id, fileSuffix=file_suffix))
        return file_path
    
    @classmethod
    def _clean_section_content(cls, content: str, section_title: str) -> str:
        """
        Kapitelinhalt bereinigen
        
        1. Markdown-Überschriftenzeilen am Inhaltsanfang entfernen, die der Kapitelüberschrift doppelt sind
        2. Alle ### und darunter liegenden Überschriften in Fetttext umwandeln
        
        Args:
            content: Originalinhalt
            section_title: Kapitelüberschrift
            
        Returns:
            Bereinigter Inhalt
        """
        import re
        
        if not content:
            return content
        
        content = content.strip()
        lines = content.split('\n')
        cleaned_lines = []
        skip_next_empty = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Prüfen, ob es eine Markdown-Überschriftenzeile ist
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            
            if heading_match:
                level = len(heading_match.group(1))
                title_text = heading_match.group(2).strip()
                
                # Prüfen, ob die Überschrift mit der Kapitelüberschrift doppelt ist (Duplikate in den ersten 5 Zeilen überspringen)
                if i < 5:
                    if title_text == section_title or title_text.replace(' ', '') == section_title.replace(' ', ''):
                        skip_next_empty = True
                        continue
                
                # Alle Überschriftenebenen (#, ##, ###, #### usw.) in Fetttext umwandeln
                # Da Kapitelüberschriften vom System hinzugefügt werden, sollte der Inhalt keine Überschriften enthalten
                cleaned_lines.append(f"**{title_text}**")
                cleaned_lines.append("")  # Leerzeile hinzufügen
                continue
            
            # Wenn vorherige Zeile eine übersprungene Überschrift war und aktuelle Zeile leer ist, ebenfalls überspringen
            if skip_next_empty and stripped == '':
                skip_next_empty = False
                continue
            
            skip_next_empty = False
            cleaned_lines.append(line)
        
        # Leerzeilen am Anfang entfernen
        while cleaned_lines and cleaned_lines[0].strip() == '':
            cleaned_lines.pop(0)
        
        # Trennlinien am Anfang entfernen
        while cleaned_lines and cleaned_lines[0].strip() in ['---', '***', '___']:
            cleaned_lines.pop(0)
            # Gleichzeitig Leerzeilen nach der Trennlinie entfernen
            while cleaned_lines and cleaned_lines[0].strip() == '':
                cleaned_lines.pop(0)
        
        return '\n'.join(cleaned_lines)
    
    @classmethod
    def update_progress(
        cls, 
        report_id: str, 
        status: str, 
        progress: int, 
        message: str,
        current_section: str = None,
        completed_sections: List[str] = None
    ) -> None:
        """
        Berichtsfortschritt aktualisieren
        
        Das Frontend kann den Echtzeitfortschritt durch Lesen von progress.json abrufen
        """
        cls._ensure_report_folder(report_id)
        
        progress_data = {
            "status": status,
            "progress": progress,
            "message": message,
            "current_section": current_section,
            "completed_sections": completed_sections or [],
            "updated_at": datetime.now().isoformat()
        }
        
        with open(cls._get_progress_path(report_id), 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def get_progress(cls, report_id: str) -> Optional[Dict[str, Any]]:
        """Berichtsfortschritt abrufen"""
        path = cls._get_progress_path(report_id)
        
        if not os.path.exists(path):
            return None
        
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @classmethod
    def get_generated_sections(cls, report_id: str) -> List[Dict[str, Any]]:
        """
        Liste der generierten Kapitel abrufen
        
        Gibt Informationen zu allen gespeicherten Kapiteldateien zurück
        """
        folder = cls._get_report_folder(report_id)
        
        if not os.path.exists(folder):
            return []
        
        sections = []
        for filename in sorted(os.listdir(folder)):
            if filename.startswith('section_') and filename.endswith('.md'):
                file_path = os.path.join(folder, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()

                # Kapitelindex aus Dateinamen parsen
                parts = filename.replace('.md', '').split('_')
                section_index = int(parts[1])

                sections.append({
                    "filename": filename,
                    "section_index": section_index,
                    "content": content
                })

        return sections
    
    @classmethod
    def assemble_full_report(cls, report_id: str, outline: ReportOutline) -> str:
        """
        Vollständigen Bericht zusammenstellen
        
        Den vollständigen Bericht aus gespeicherten Kapiteldateien zusammenstellen und Überschriften bereinigen
        """
        folder = cls._get_report_folder(report_id)
        
        # Berichtskopf erstellen
        md_content = f"# {outline.title}\n\n"
        md_content += f"> {outline.summary}\n\n"
        md_content += f"---\n\n"
        
        # Alle Kapiteldateien in Reihenfolge lesen
        sections = cls.get_generated_sections(report_id)
        for section_info in sections:
            md_content += section_info["content"]
        
        # Nachbearbeitung: Überschriftenprobleme im gesamten Bericht bereinigen
        md_content = cls._post_process_report(md_content, outline)
        
        # Vollständigen Bericht speichern
        full_path = cls._get_report_markdown_path(report_id)
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        logger.info(t('report.fullReportAssembled', reportId=report_id))
        return md_content
    
    @classmethod
    def _post_process_report(cls, content: str, outline: ReportOutline) -> str:
        """
        Berichtsinhalt nachbearbeiten
        
        1. Doppelte Überschriften entfernen
        2. Berichtshauptüberschrift (#) und Kapitelüberschriften (##) beibehalten, andere Ebenen (###, #### usw.) entfernen
        3. Überflüssige Leerzeilen und Trennlinien bereinigen
        
        Args:
            content: Originaler Berichtsinhalt
            outline: Berichtsgliederung
            
        Returns:
            Verarbeiteter Inhalt
        """
        import re
        
        lines = content.split('\n')
        processed_lines = []
        prev_was_heading = False
        
        # Alle Kapitelüberschriften aus der Gliederung sammeln
        section_titles = set()
        for section in outline.sections:
            section_titles.add(section.title)
        
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()
            
            # Prüfen, ob es eine Überschriftenzeile ist
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', stripped)
            
            if heading_match:
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                
                # Prüfen, ob es eine doppelte Überschrift ist (gleicher Inhalt innerhalb von 5 aufeinanderfolgenden Zeilen)
                is_duplicate = False
                for j in range(max(0, len(processed_lines) - 5), len(processed_lines)):
                    prev_line = processed_lines[j].strip()
                    prev_match = re.match(r'^(#{1,6})\s+(.+)$', prev_line)
                    if prev_match:
                        prev_title = prev_match.group(2).strip()
                        if prev_title == title:
                            is_duplicate = True
                            break
                
                if is_duplicate:
                    # Doppelte Überschrift und nachfolgende Leerzeilen überspringen
                    i += 1
                    while i < len(lines) and lines[i].strip() == '':
                        i += 1
                    continue
                
                # Überschriften-Ebenenbehandlung:
                # - # (level=1) Nur Berichtshauptüberschrift beibehalten
                # - ## (level=2) Kapitelüberschriften beibehalten
                # - ### und darunter (level>=3) In Fetttext umwandeln
                
                if level == 1:
                    if title == outline.title:
                        # Berichtshauptüberschrift beibehalten
                        processed_lines.append(line)
                        prev_was_heading = True
                    elif title in section_titles:
                        # Kapitelüberschrift hat fälschlicherweise # verwendet, zu ## korrigieren
                        processed_lines.append(f"## {title}")
                        prev_was_heading = True
                    else:
                        # Andere Ebene-1-Überschriften in Fetttext umwandeln
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                elif level == 2:
                    if title in section_titles or title == outline.title:
                        # Kapitelüberschrift beibehalten
                        processed_lines.append(line)
                        prev_was_heading = True
                    else:
                        # Nicht-Kapitel-Ebene-2-Überschriften in Fetttext umwandeln
                        processed_lines.append(f"**{title}**")
                        processed_lines.append("")
                        prev_was_heading = False
                else:
                    # ### und darunter liegende Überschriften in Fetttext umwandeln
                    processed_lines.append(f"**{title}**")
                    processed_lines.append("")
                    prev_was_heading = False
                
                i += 1
                continue
            
            elif stripped == '---' and prev_was_heading:
                # Trennlinie direkt nach Überschrift überspringen
                i += 1
                continue
            
            elif stripped == '' and prev_was_heading:
                # Nur eine Leerzeile nach der Überschrift beibehalten
                if processed_lines and processed_lines[-1].strip() != '':
                    processed_lines.append(line)
                prev_was_heading = False
            
            else:
                processed_lines.append(line)
                prev_was_heading = False
            
            i += 1
        
        # Aufeinanderfolgende Leerzeilen bereinigen (maximal 2 beibehalten)
        result_lines = []
        empty_count = 0
        for line in processed_lines:
            if line.strip() == '':
                empty_count += 1
                if empty_count <= 2:
                    result_lines.append(line)
            else:
                empty_count = 0
                result_lines.append(line)
        
        return '\n'.join(result_lines)
    
    @classmethod
    def save_report(cls, report: Report) -> None:
        """Bericht-Metainformationen und vollständigen Bericht speichern"""
        cls._ensure_report_folder(report.report_id)
        
        # Metainformationen-JSON speichern
        with open(cls._get_report_path(report.report_id), 'w', encoding='utf-8') as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        
        # Gliederung speichern
        if report.outline:
            cls.save_outline(report.report_id, report.outline)
        
        # Vollständigen Markdown-Bericht speichern
        if report.markdown_content:
            with open(cls._get_report_markdown_path(report.report_id), 'w', encoding='utf-8') as f:
                f.write(report.markdown_content)
        
        logger.info(t('report.reportSaved', reportId=report.report_id))
    
    @classmethod
    def get_report(cls, report_id: str) -> Optional[Report]:
        """Bericht abrufen"""
        path = cls._get_report_path(report_id)
        
        if not os.path.exists(path):
            # Abwärtskompatibilität: Dateien prüfen, die direkt im reports-Verzeichnis gespeichert sind
            old_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
            if os.path.exists(old_path):
                path = old_path
            else:
                return None
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Report-Objekt rekonstruieren
        outline = None
        if data.get('outline'):
            outline_data = data['outline']
            sections = []
            for s in outline_data.get('sections', []):
                sections.append(ReportSection(
                    title=s['title'],
                    content=s.get('content', '')
                ))
            outline = ReportOutline(
                title=outline_data['title'],
                summary=outline_data['summary'],
                sections=sections
            )
        
        # Wenn markdown_content leer ist, versuchen aus full_report.md zu lesen
        markdown_content = data.get('markdown_content', '')
        if not markdown_content:
            full_report_path = cls._get_report_markdown_path(report_id)
            if os.path.exists(full_report_path):
                with open(full_report_path, 'r', encoding='utf-8') as f:
                    markdown_content = f.read()
        
        return Report(
            report_id=data['report_id'],
            simulation_id=data['simulation_id'],
            graph_id=data['graph_id'],
            simulation_requirement=data['simulation_requirement'],
            status=ReportStatus(data['status']),
            outline=outline,
            markdown_content=markdown_content,
            created_at=data.get('created_at', ''),
            completed_at=data.get('completed_at', ''),
            error=data.get('error')
        )
    
    @classmethod
    def get_report_by_simulation(cls, simulation_id: str) -> Optional[Report]:
        """Bericht anhand der Simulations-ID abrufen"""
        cls._ensure_reports_dir()
        
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # Neues Format: Ordner
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report and report.simulation_id == simulation_id:
                    return report
            # Abwärtskompatibilität: JSON-Datei
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report and report.simulation_id == simulation_id:
                    return report
        
        return None
    
    @classmethod
    def list_reports(cls, simulation_id: Optional[str] = None, limit: int = 50) -> List[Report]:
        """Berichte auflisten"""
        cls._ensure_reports_dir()
        
        reports = []
        for item in os.listdir(cls.REPORTS_DIR):
            item_path = os.path.join(cls.REPORTS_DIR, item)
            # Neues Format: Ordner
            if os.path.isdir(item_path):
                report = cls.get_report(item)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
            # Abwärtskompatibilität: JSON-Datei
            elif item.endswith('.json'):
                report_id = item[:-5]
                report = cls.get_report(report_id)
                if report:
                    if simulation_id is None or report.simulation_id == simulation_id:
                        reports.append(report)
        
        # Nach Erstellungszeit absteigend sortieren
        reports.sort(key=lambda r: r.created_at, reverse=True)
        
        return reports[:limit]
    
    @classmethod
    def delete_report(cls, report_id: str) -> bool:
        """Bericht löschen (gesamter Ordner)"""
        import shutil
        
        folder_path = cls._get_report_folder(report_id)
        
        # Neues Format: Gesamten Ordner löschen
        if os.path.exists(folder_path) and os.path.isdir(folder_path):
            shutil.rmtree(folder_path)
            logger.info(t('report.reportFolderDeleted', reportId=report_id))
            return True
        
        # Abwärtskompatibilität: Einzelne Dateien löschen
        deleted = False
        old_json_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.json")
        old_md_path = os.path.join(cls.REPORTS_DIR, f"{report_id}.md")
        
        if os.path.exists(old_json_path):
            os.remove(old_json_path)
            deleted = True
        if os.path.exists(old_md_path):
            os.remove(old_md_path)
            deleted = True
        
        return deleted
