"""
Intelligenter Generator für Simulationskonfigurationen
Verwendet LLM, um anhand von Simulationsanforderungen, Dokumentinhalten und Graphinformationen automatisch detaillierte Simulationsparameter zu generieren
Vollständig automatisiert, keine manuelle Parametereinstellung erforderlich

Schrittweise Generierungsstrategie, um Fehler durch zu lange Einzelgenerierung zu vermeiden:
1. Zeitkonfiguration generieren
2. Ereigniskonfiguration generieren
3. Agent-Konfigurationen in Batches generieren
4. Plattformkonfiguration generieren
"""

import json
import math
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime

from openai import OpenAI

from ..config import Config
from ..utils.logger import get_logger
from ..utils.locale import get_language_instruction, t
from .zep_entity_reader import EntityNode, ZepEntityReader

logger = get_logger('mirofish.simulation_config')

# Konfiguration der chinesischen Tagesrhythmen (Pekinger Zeit)
CHINA_TIMEZONE_CONFIG = {
    # Nachtzeit (fast keine Aktivität)
    "dead_hours": [0, 1, 2, 3, 4, 5],
    # Morgenstunden (allmähliches Aufwachen)
    "morning_hours": [6, 7, 8],
    # Arbeitszeit
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    # Abendliche Spitzenzeit (am aktivsten)
    "peak_hours": [19, 20, 21, 22],
    # Nachtstunden (Aktivität nimmt ab)
    "night_hours": [23],
    # Aktivitätskoeffizienten
    "activity_multipliers": {
        "dead": 0.05,      # Frühe Morgenstunden, fast niemand aktiv
        "morning": 0.4,    # Morgens zunehmend aktiv
        "work": 0.7,       # Arbeitszeit mittelmäßig
        "peak": 1.5,       # Abendliche Spitzenzeit
        "night": 0.5       # Spätabends abnehmend
    }
}


@dataclass
class AgentActivityConfig:
    """Aktivitätskonfiguration eines einzelnen Agents"""
    agent_id: int
    entity_uuid: str
    entity_name: str
    entity_type: str
    
    # Aktivitätskonfiguration (0.0-1.0)
    activity_level: float = 0.5  # Gesamtaktivität
    
    # Beitragshäufigkeit (erwartete Beiträge pro Stunde)
    posts_per_hour: float = 1.0
    comments_per_hour: float = 2.0
    
    # Aktive Zeiträume (24-Stunden-Format, 0-23)
    active_hours: List[int] = field(default_factory=lambda: list(range(8, 23)))
    
    # Reaktionsgeschwindigkeit (Verzögerung bei Trendthemen, Einheit: Simulationsminuten)
    response_delay_min: int = 5
    response_delay_max: int = 60
    
    # Stimmungstendenz (-1.0 bis 1.0, negativ bis positiv)
    sentiment_bias: float = 0.0
    
    # Standpunkt (Haltung zu bestimmten Themen)
    stance: str = "neutral"  # supportive, opposing, neutral, observer
    
    # Einflussgewichtung (bestimmt die Wahrscheinlichkeit, dass Beiträge von anderen Agents gesehen werden)
    influence_weight: float = 1.0


@dataclass  
class TimeSimulationConfig:
    """Zeitsimulationskonfiguration (basierend auf chinesischen Tagesrhythmen)"""
    # Gesamtdauer der Simulation (Simulationsstunden)
    total_simulation_hours: int = 72  # Standardmäßig 72 Stunden (3 Tage) simulieren
    
    # Zeitdauer pro Runde (Simulationsminuten) - Standard 60 Minuten (1 Stunde), beschleunigter Zeitfluss
    minutes_per_round: int = 60
    
    # Bereich der pro Stunde aktivierten Agents
    agents_per_hour_min: int = 5
    agents_per_hour_max: int = 20
    
    # Spitzenzeiten (abends 19-22 Uhr, aktivste Zeit für chinesische Nutzer)
    peak_hours: List[int] = field(default_factory=lambda: [19, 20, 21, 22])
    peak_activity_multiplier: float = 1.5
    
    # Tiefpunktzeiten (0-5 Uhr morgens, fast keine Aktivität)
    off_peak_hours: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5])
    off_peak_activity_multiplier: float = 0.05  # Sehr geringe Aktivität in den frühen Morgenstunden
    
    # Morgenstunden
    morning_hours: List[int] = field(default_factory=lambda: [6, 7, 8])
    morning_activity_multiplier: float = 0.4
    
    # Arbeitszeit
    work_hours: List[int] = field(default_factory=lambda: [9, 10, 11, 12, 13, 14, 15, 16, 17, 18])
    work_activity_multiplier: float = 0.7


@dataclass
class EventConfig:
    """Ereigniskonfiguration"""
    # Anfangsereignisse (Auslöseereignisse bei Simulationsstart)
    initial_posts: List[Dict[str, Any]] = field(default_factory=list)
    
    # Geplante Ereignisse (zu bestimmten Zeiten ausgelöste Ereignisse)
    scheduled_events: List[Dict[str, Any]] = field(default_factory=list)
    
    # Schlüsselwörter für Trendthemen
    hot_topics: List[str] = field(default_factory=list)
    
    # Richtung der Meinungsführung
    narrative_direction: str = ""


@dataclass
class PlatformConfig:
    """Plattformspezifische Konfiguration"""
    platform: str  # twitter or reddit
    
    # Gewichtung des Empfehlungsalgorithmus
    recency_weight: float = 0.4  # Zeitliche Aktualität
    popularity_weight: float = 0.3  # Beliebtheit
    relevance_weight: float = 0.3  # Relevanz
    
    # Virale Verbreitungsschwelle (ab wie vielen Interaktionen Verbreitung ausgelöst wird)
    viral_threshold: int = 10
    
    # Stärke des Echokammer-Effekts (Grad der Clusterbildung ähnlicher Meinungen)
    echo_chamber_strength: float = 0.5


@dataclass
class SimulationParameters:
    """Vollständige Simulationsparameterkonfiguration"""
    # Basisinformationen
    simulation_id: str
    project_id: str
    graph_id: str
    simulation_requirement: str
    
    # Zeitkonfiguration
    time_config: TimeSimulationConfig = field(default_factory=TimeSimulationConfig)
    
    # Agent-Konfigurationsliste
    agent_configs: List[AgentActivityConfig] = field(default_factory=list)
    
    # Ereigniskonfiguration
    event_config: EventConfig = field(default_factory=EventConfig)
    
    # Plattformkonfiguration
    twitter_config: Optional[PlatformConfig] = None
    reddit_config: Optional[PlatformConfig] = None
    
    # LLM-Konfiguration
    llm_model: str = ""
    llm_base_url: str = ""
    
    # Generierungsmetadaten
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    generation_reasoning: str = ""  # Erläuterung der LLM-Argumentation
    
    def to_dict(self) -> Dict[str, Any]:
        """In ein Dictionary umwandeln"""
        time_dict = asdict(self.time_config)
        return {
            "simulation_id": self.simulation_id,
            "project_id": self.project_id,
            "graph_id": self.graph_id,
            "simulation_requirement": self.simulation_requirement,
            "time_config": time_dict,
            "agent_configs": [asdict(a) for a in self.agent_configs],
            "event_config": asdict(self.event_config),
            "twitter_config": asdict(self.twitter_config) if self.twitter_config else None,
            "reddit_config": asdict(self.reddit_config) if self.reddit_config else None,
            "llm_model": self.llm_model,
            "llm_base_url": self.llm_base_url,
            "generated_at": self.generated_at,
            "generation_reasoning": self.generation_reasoning,
        }
    
    def to_json(self, indent: int = 2) -> str:
        """In einen JSON-String umwandeln"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class SimulationConfigGenerator:
    """
    Intelligenter Simulationskonfigurations-Generator
    
    Verwendet LLM zur Analyse von Simulationsanforderungen, Dokumentinhalten und Graph-Entitätsinformationen,
    um automatisch die optimale Simulationsparameterkonfiguration zu generieren
    
    Schrittweise Generierungsstrategie:
    1. Zeit- und Ereigniskonfiguration generieren (leichtgewichtig)
    2. Agent-Konfigurationen in Batches generieren (10-20 pro Batch)
    3. Plattformkonfiguration generieren
    """
    
    # Maximale Zeichenanzahl des Kontexts
    MAX_CONTEXT_LENGTH = 50000
    # Anzahl der Agents pro Batch
    AGENTS_PER_BATCH = 15
    
    # Kontextkürzungslänge pro Schritt (Zeichenanzahl)
    TIME_CONFIG_CONTEXT_LENGTH = 10000   # Zeitkonfiguration
    EVENT_CONFIG_CONTEXT_LENGTH = 8000   # Ereigniskonfiguration
    ENTITY_SUMMARY_LENGTH = 300          # Entitätszusammenfassung
    AGENT_SUMMARY_LENGTH = 300           # Entitätszusammenfassung in der Agent-Konfiguration
    ENTITIES_PER_TYPE_DISPLAY = 20       # Anzahl angezeigter Entitäten pro Typ
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.api_key = api_key or Config.LLM_API_KEY
        self.base_url = base_url or Config.LLM_BASE_URL
        self.model_name = model_name or Config.LLM_MODEL_NAME
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY nicht konfiguriert")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def generate_config(
        self,
        simulation_id: str,
        project_id: str,
        graph_id: str,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode],
        enable_twitter: bool = True,
        enable_reddit: bool = True,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> SimulationParameters:
        """
        Intelligente Generierung einer vollständigen Simulationskonfiguration (schrittweise Generierung)
        
        Args:
            simulation_id: Simulations-ID
            project_id: Projekt-ID
            graph_id: Graph-ID
            simulation_requirement: Beschreibung der Simulationsanforderungen
            document_text: Originaler Dokumentinhalt
            entities: Gefilterte Entitätenliste
            enable_twitter: Ob Twitter aktiviert werden soll
            enable_reddit: Ob Reddit aktiviert werden soll
            progress_callback: Fortschritts-Callback-Funktion(current_step, total_steps, message)
            
        Returns:
            SimulationParameters: Vollständige Simulationsparameter
        """
        logger.info(f"Starte intelligente Generierung der Simulationskonfiguration: simulation_id={simulation_id}, Entitätenanzahl={len(entities)}")
        
        # Gesamtschrittanzahl berechnen
        num_batches = math.ceil(len(entities) / self.AGENTS_PER_BATCH)
        total_steps = 3 + num_batches  # Zeitkonfiguration + Ereigniskonfiguration + N Agent-Batches + Plattformkonfiguration
        current_step = 0
        
        def report_progress(step: int, message: str):
            nonlocal current_step
            current_step = step
            if progress_callback:
                progress_callback(step, total_steps, message)
            logger.info(f"[{step}/{total_steps}] {message}")
        
        # 1. Basiskontextinformationen aufbauen
        context = self._build_context(
            simulation_requirement=simulation_requirement,
            document_text=document_text,
            entities=entities
        )
        
        reasoning_parts = []
        
        # ========== Schritt 1: Zeitkonfiguration generieren ==========
        report_progress(1, t('progress.generatingTimeConfig'))
        num_entities = len(entities)
        time_config_result = self._generate_time_config(context, num_entities)
        time_config = self._parse_time_config(time_config_result, num_entities)
        reasoning_parts.append(f"{t('progress.timeConfigLabel')}: {time_config_result.get('reasoning', t('common.success'))}")
        
        # ========== Schritt 2: Ereigniskonfiguration generieren ==========
        report_progress(2, t('progress.generatingEventConfig'))
        event_config_result = self._generate_event_config(context, simulation_requirement, entities)
        event_config = self._parse_event_config(event_config_result)
        reasoning_parts.append(f"{t('progress.eventConfigLabel')}: {event_config_result.get('reasoning', t('common.success'))}")
        
        # ========== Schritt 3-N: Agent-Konfigurationen in Batches generieren ==========
        all_agent_configs = []
        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.AGENTS_PER_BATCH
            end_idx = min(start_idx + self.AGENTS_PER_BATCH, len(entities))
            batch_entities = entities[start_idx:end_idx]
            
            report_progress(
                3 + batch_idx,
                t('progress.generatingAgentConfig', start=start_idx + 1, end=end_idx, total=len(entities))
            )
            
            batch_configs = self._generate_agent_configs_batch(
                context=context,
                entities=batch_entities,
                start_idx=start_idx,
                simulation_requirement=simulation_requirement
            )
            all_agent_configs.extend(batch_configs)
        
        reasoning_parts.append(t('progress.agentConfigResult', count=len(all_agent_configs)))
        
        # ========== Anfangsbeiträgen Veröffentlicher-Agents zuweisen ==========
        logger.info("Geeignete Veröffentlicher-Agents für Anfangsbeiträge zuweisen...")
        event_config = self._assign_initial_post_agents(event_config, all_agent_configs)
        assigned_count = len([p for p in event_config.initial_posts if p.get("poster_agent_id") is not None])
        reasoning_parts.append(t('progress.postAssignResult', count=assigned_count))
        
        # ========== Letzter Schritt: Plattformkonfiguration generieren ==========
        report_progress(total_steps, t('progress.generatingPlatformConfig'))
        twitter_config = None
        reddit_config = None
        
        if enable_twitter:
            twitter_config = PlatformConfig(
                platform="twitter",
                recency_weight=0.4,
                popularity_weight=0.3,
                relevance_weight=0.3,
                viral_threshold=10,
                echo_chamber_strength=0.5
            )
        
        if enable_reddit:
            reddit_config = PlatformConfig(
                platform="reddit",
                recency_weight=0.3,
                popularity_weight=0.4,
                relevance_weight=0.3,
                viral_threshold=15,
                echo_chamber_strength=0.6
            )
        
        # Endgültige Parameter erstellen
        params = SimulationParameters(
            simulation_id=simulation_id,
            project_id=project_id,
            graph_id=graph_id,
            simulation_requirement=simulation_requirement,
            time_config=time_config,
            agent_configs=all_agent_configs,
            event_config=event_config,
            twitter_config=twitter_config,
            reddit_config=reddit_config,
            llm_model=self.model_name,
            llm_base_url=self.base_url,
            generation_reasoning=" | ".join(reasoning_parts)
        )
        
        logger.info(f"Simulationskonfiguration vollständig generiert: {len(params.agent_configs)} Agent-Konfigurationen")
        
        return params
    
    def _build_context(
        self,
        simulation_requirement: str,
        document_text: str,
        entities: List[EntityNode]
    ) -> str:
        """LLM-Kontext aufbauen, auf maximale Länge kürzen"""
        
        # Entitätszusammenfassung
        entity_summary = self._summarize_entities(entities)
        
        # Kontext aufbauen
        context_parts = [
            f"## Simulationsanforderungen\n{simulation_requirement}",
            f"\n## Entitätsinformationen ({len(entities)} Stück)\n{entity_summary}",
        ]
        
        current_length = sum(len(p) for p in context_parts)
        remaining_length = self.MAX_CONTEXT_LENGTH - current_length - 500  # 500 Zeichen Puffer
        
        if remaining_length > 0 and document_text:
            doc_text = document_text[:remaining_length]
            if len(document_text) > remaining_length:
                doc_text += "\n...(Dokument wurde gekürzt)"
            context_parts.append(f"\n## Originaler Dokumentinhalt\n{doc_text}")
        
        return "\n".join(context_parts)
    
    def _summarize_entities(self, entities: List[EntityNode]) -> str:
        """Entitätszusammenfassung generieren"""
        lines = []
        
        # Nach Typ gruppieren
        by_type: Dict[str, List[EntityNode]] = {}
        for e in entities:
            t = e.get_entity_type() or "Unknown"
            if t not in by_type:
                by_type[t] = []
            by_type[t].append(e)
        
        for entity_type, type_entities in by_type.items():
            lines.append(f"\n### {entity_type} ({len(type_entities)} Stück)")
            # Konfigurierte Anzeigeanzahl und Zusammenfassungslänge verwenden
            display_count = self.ENTITIES_PER_TYPE_DISPLAY
            summary_len = self.ENTITY_SUMMARY_LENGTH
            for e in type_entities[:display_count]:
                summary_preview = (e.summary[:summary_len] + "...") if len(e.summary) > summary_len else e.summary
                lines.append(f"- {e.name}: {summary_preview}")
            if len(type_entities) > display_count:
                lines.append(f"  ... und {len(type_entities) - display_count} weitere")
        
        return "\n".join(lines)
    
    def _call_llm_with_retry(self, prompt: str, system_prompt: str) -> Dict[str, Any]:
        """LLM-Aufruf mit Wiederholung, einschließlich JSON-Reparaturlogik"""
        import re
        
        max_attempts = 3
        last_error = None
        
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.7 - (attempt * 0.1)  # Temperatur bei jedem Versuch senken
                    # max_tokens nicht setzen, LLM frei generieren lassen
                )
                
                content = response.choices[0].message.content
                finish_reason = response.choices[0].finish_reason
                
                # Prüfen ob abgeschnitten
                if finish_reason == 'length':
                    logger.warning(f"LLM-Ausgabe wurde abgeschnitten (attempt {attempt+1})")
                    content = self._fix_truncated_json(content)
                
                # JSON-Parsing versuchen
                try:
                    return json.loads(content)
                except json.JSONDecodeError as e:
                    logger.warning(f"JSON-Parsing fehlgeschlagen (attempt {attempt+1}): {str(e)[:80]}")
                    
                    # JSON-Reparatur versuchen
                    fixed = self._try_fix_config_json(content)
                    if fixed:
                        return fixed
                    
                    last_error = e
                    
            except Exception as e:
                logger.warning(f"LLM-Aufruf fehlgeschlagen (attempt {attempt+1}): {str(e)[:80]}")
                last_error = e
                import time
                time.sleep(2 * (attempt + 1))
        
        raise last_error or Exception("LLM-Aufruf fehlgeschlagen")
    
    def _fix_truncated_json(self, content: str) -> str:
        """Abgeschnittenes JSON reparieren"""
        content = content.strip()
        
        # Nicht geschlossene Klammern zählen
        open_braces = content.count('{') - content.count('}')
        open_brackets = content.count('[') - content.count(']')
        
        # Prüfen ob nicht geschlossene Zeichenketten vorhanden
        if content and content[-1] not in '",}]':
            content += '"'
        
        # Klammern schließen
        content += ']' * open_brackets
        content += '}' * open_braces
        
        return content
    
    def _try_fix_config_json(self, content: str) -> Optional[Dict[str, Any]]:
        """Konfiguration-JSON reparieren versuchen"""
        import re
        
        # Abgeschnittene Situation reparieren
        content = self._fix_truncated_json(content)
        
        # JSON-Teil extrahieren
        json_match = re.search(r'\{[\s\S]*\}', content)
        if json_match:
            json_str = json_match.group()
            
            # Zeilenumbrüche in Zeichenketten entfernen
            def fix_string(match):
                s = match.group(0)
                s = s.replace('\n', ' ').replace('\r', ' ')
                s = re.sub(r'\s+', ' ', s)
                return s
            
            json_str = re.sub(r'"[^"\\]*(?:\\.[^"\\]*)*"', fix_string, json_str)
            
            try:
                return json.loads(json_str)
            except:
                # Alle Steuerzeichen entfernen versuchen
                json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', json_str)
                json_str = re.sub(r'\s+', ' ', json_str)
                try:
                    return json.loads(json_str)
                except:
                    pass
        
        return None
    
    def _generate_time_config(self, context: str, num_entities: int) -> Dict[str, Any]:
        """Zeitkonfiguration generieren"""
        # Konfigurierte Kontextkürzungslänge verwenden
        context_truncated = context[:self.TIME_CONFIG_CONTEXT_LENGTH]
        
        # Maximal erlaubten Wert berechnen (80% der Agents)
        max_agents_allowed = max(1, int(num_entities * 0.9))
        
        prompt = f"""Basierend auf den folgenden Simulationsanforderungen, generiere eine Zeitsimulationskonfiguration.

{context_truncated}

## Aufgabe
Bitte generiere eine Zeitkonfigurations-JSON.

### Grundprinzipien (nur als Referenz, sollte je nach konkretem Ereignis und Zielgruppe flexibel angepasst werden):
- Bitte leite die Zeitzone und Tagesrhythmen der Zielnutzergruppe aus dem Simulationsszenario ab. Folgendes ist ein Referenzbeispiel für UTC+8
- 0-5 Uhr morgens fast keine Aktivität (Aktivitätskoeffizient 0,05)
- 6-8 Uhr morgens zunehmend aktiv (Aktivitätskoeffizient 0,4)
- Arbeitszeit 9-18 Uhr mittelmäßig aktiv (Aktivitätskoeffizient 0,7)
- Abends 19-22 Uhr ist Spitzenzeit (Aktivitätskoeffizient 1,5)
- Nach 23 Uhr nimmt die Aktivität ab (Aktivitätskoeffizient 0,5)
- Allgemeine Regel: niedrige Aktivität in den frühen Morgenstunden, morgens zunehmend, mittelmäßig während der Arbeitszeit, abendliche Spitze
- **Wichtig**: Die folgenden Beispielwerte dienen nur als Referenz. Sie müssen die konkreten Zeiträume je nach Art des Ereignisses und Merkmalen der Zielgruppe anpassen
  - Beispiel: Studenten haben möglicherweise ihre Spitzenzeit von 21-23 Uhr; Medien sind ganztägig aktiv; offizielle Stellen nur während der Arbeitszeit
  - Beispiel: Aktuelle Trendthemen können zu Diskussionen auch in der Nacht führen, off_peak_hours können entsprechend verkürzt werden

### JSON-Rückgabeformat (kein Markdown)

Beispiel:
{{
    "total_simulation_hours": 72,
    "minutes_per_round": 60,
    "agents_per_hour_min": 5,
    "agents_per_hour_max": 50,
    "peak_hours": [19, 20, 21, 22],
    "off_peak_hours": [0, 1, 2, 3, 4, 5],
    "morning_hours": [6, 7, 8],
    "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
    "reasoning": "Erläuterung der Zeitkonfiguration für dieses Ereignis"
}}

Feldbeschreibungen:
- total_simulation_hours (int): Gesamtdauer der Simulation, 24-168 Stunden, kurz für plötzliche Ereignisse, lang für anhaltende Themen
- minutes_per_round (int): Dauer pro Runde, 30-120 Minuten, empfohlen 60 Minuten
- agents_per_hour_min (int): Mindestanzahl aktivierter Agents pro Stunde (Wertebereich: 1-{max_agents_allowed})
- agents_per_hour_max (int): Höchstanzahl aktivierter Agents pro Stunde (Wertebereich: 1-{max_agents_allowed})
- peak_hours (int-Array): Spitzenzeiten, je nach Zielgruppe des Ereignisses anpassen
- off_peak_hours (int-Array): Tiefpunktzeiten, normalerweise späte Nacht/früher Morgen
- morning_hours (int-Array): Morgenstunden
- work_hours (int-Array): Arbeitszeiten
- reasoning (string): Kurze Erläuterung, warum diese Konfiguration gewählt wurde"""

        system_prompt = "Du bist ein Experte für Social-Media-Simulationen. Gib reines JSON-Format zurück. Die Zeitkonfiguration muss den Tagesrhythmen der Zielnutzergruppe im Simulationsszenario entsprechen."
        system_prompt = f"{system_prompt}\n\n{get_language_instruction()}"

        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(f"LLM-Generierung der Zeitkonfiguration fehlgeschlagen: {e}, verwende Standardkonfiguration")
            return self._get_default_time_config(num_entities)
    
    def _get_default_time_config(self, num_entities: int) -> Dict[str, Any]:
        """Standard-Zeitkonfiguration abrufen (chinesische Tagesrhythmen)"""
        return {
            "total_simulation_hours": 72,
            "minutes_per_round": 60,  # 1 Stunde pro Runde, beschleunigter Zeitfluss
            "agents_per_hour_min": max(1, num_entities // 15),
            "agents_per_hour_max": max(5, num_entities // 5),
            "peak_hours": [19, 20, 21, 22],
            "off_peak_hours": [0, 1, 2, 3, 4, 5],
            "morning_hours": [6, 7, 8],
            "work_hours": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
            "reasoning": "Standard-Konfiguration für chinesische Tagesrhythmen (1 Stunde pro Runde)"
        }
    
    def _parse_time_config(self, result: Dict[str, Any], num_entities: int) -> TimeSimulationConfig:
        """Zeitkonfigurationsergebnis parsen und überprüfen, dass agents_per_hour-Werte die Gesamt-Agent-Anzahl nicht überschreiten"""
        # Ursprüngliche Werte abrufen
        agents_per_hour_min = result.get("agents_per_hour_min", max(1, num_entities // 15))
        agents_per_hour_max = result.get("agents_per_hour_max", max(5, num_entities // 5))
        
        # Überprüfen und korrigieren: Sicherstellen, dass Gesamt-Agent-Anzahl nicht überschritten wird
        if agents_per_hour_min > num_entities:
            logger.warning(f"agents_per_hour_min ({agents_per_hour_min}) übersteigt Gesamt-Agent-Anzahl ({num_entities}), wurde korrigiert")
            agents_per_hour_min = max(1, num_entities // 10)
        
        if agents_per_hour_max > num_entities:
            logger.warning(f"agents_per_hour_max ({agents_per_hour_max}) übersteigt Gesamt-Agent-Anzahl ({num_entities}), wurde korrigiert")
            agents_per_hour_max = max(agents_per_hour_min + 1, num_entities // 2)
        
        # Sicherstellen, dass min < max
        if agents_per_hour_min >= agents_per_hour_max:
            agents_per_hour_min = max(1, agents_per_hour_max // 2)
            logger.warning(f"agents_per_hour_min >= max, korrigiert auf {agents_per_hour_min}")
        
        return TimeSimulationConfig(
            total_simulation_hours=result.get("total_simulation_hours", 72),
            minutes_per_round=result.get("minutes_per_round", 60),  # Standard: 1 Stunde pro Runde
            agents_per_hour_min=agents_per_hour_min,
            agents_per_hour_max=agents_per_hour_max,
            peak_hours=result.get("peak_hours", [19, 20, 21, 22]),
            off_peak_hours=result.get("off_peak_hours", [0, 1, 2, 3, 4, 5]),
            off_peak_activity_multiplier=0.05,  # Frühe Morgenstunden, fast niemand aktiv
            morning_hours=result.get("morning_hours", [6, 7, 8]),
            morning_activity_multiplier=0.4,
            work_hours=result.get("work_hours", list(range(9, 19))),
            work_activity_multiplier=0.7,
            peak_activity_multiplier=1.5
        )
    
    def _generate_event_config(
        self, 
        context: str, 
        simulation_requirement: str,
        entities: List[EntityNode]
    ) -> Dict[str, Any]:
        """Ereigniskonfiguration generieren"""
        
        # Verfügbare Entitätstypen-Liste abrufen, als Referenz für LLM
        entity_types_available = list(set(
            e.get_entity_type() or "Unknown" for e in entities
        ))
        
        # Repräsentative Entitätsnamen für jeden Typ auflisten
        type_examples = {}
        for e in entities:
            etype = e.get_entity_type() or "Unknown"
            if etype not in type_examples:
                type_examples[etype] = []
            if len(type_examples[etype]) < 3:
                type_examples[etype].append(e.name)
        
        type_info = "\n".join([
            f"- {t}: {', '.join(examples)}" 
            for t, examples in type_examples.items()
        ])
        
        # Konfigurierte Kontextkürzungslänge verwenden
        context_truncated = context[:self.EVENT_CONFIG_CONTEXT_LENGTH]
        
        prompt = f"""Basierend auf den folgenden Simulationsanforderungen, generiere eine Ereigniskonfiguration.

Simulationsanforderungen: {simulation_requirement}

{context_truncated}

## Verfügbare Entitätstypen und Beispiele
{type_info}

## Aufgabe
Bitte generiere eine Ereigniskonfigurations-JSON:
- Schlüsselwörter für Trendthemen extrahieren
- Richtung der Meinungsentwicklung beschreiben
- Anfangsbeiträge gestalten, **jeder Beitrag muss einen poster_type (Veröffentlichertyp) angeben**

**Wichtig**: poster_type muss aus den oben aufgeführten "Verfügbaren Entitätstypen" ausgewählt werden, damit Anfangsbeiträge geeigneten Agents zugewiesen werden können.
Beispiel: Offizielle Erklärungen sollten vom Typ Official/University veröffentlicht werden, Nachrichten von MediaOutlet, Studentenmeinungen von Student.

JSON-Rückgabeformat (kein Markdown):
{{
    "hot_topics": ["Schlüsselwort1", "Schlüsselwort2", ...],
    "narrative_direction": "<Beschreibung der Meinungsentwicklung>",
    "initial_posts": [
        {{"content": "Beitragsinhalt", "poster_type": "Entitätstyp (muss aus verfügbaren Typen ausgewählt werden)"}},
        ...
    ],
    "reasoning": "<Kurze Erläuterung>"
}}"""

        system_prompt = "Du bist ein Experte für Meinungsanalyse. Gib reines JSON-Format zurück. Beachte, dass poster_type genau mit den verfügbaren Entitätstypen übereinstimmen muss."
        system_prompt = f"{system_prompt}\n\n{get_language_instruction()}\nIMPORTANT: The 'poster_type' field value MUST be in English PascalCase exactly matching the available entity types. Only 'content', 'narrative_direction', 'hot_topics' and 'reasoning' fields should use the specified language."

        try:
            return self._call_llm_with_retry(prompt, system_prompt)
        except Exception as e:
            logger.warning(f"LLM-Generierung der Ereigniskonfiguration fehlgeschlagen: {e}, verwende Standardkonfiguration")
            return {
                "hot_topics": [],
                "narrative_direction": "",
                "initial_posts": [],
                "reasoning": "Standardkonfiguration verwendet"
            }
    
    def _parse_event_config(self, result: Dict[str, Any]) -> EventConfig:
        """Ereigniskonfigurationsergebnis parsen"""
        return EventConfig(
            initial_posts=result.get("initial_posts", []),
            scheduled_events=[],
            hot_topics=result.get("hot_topics", []),
            narrative_direction=result.get("narrative_direction", "")
        )
    
    def _assign_initial_post_agents(
        self,
        event_config: EventConfig,
        agent_configs: List[AgentActivityConfig]
    ) -> EventConfig:
        """
        Geeignete Veröffentlicher-Agents für Anfangsbeiträge zuweisen
        
        Für jeden Beitrag die passendste agent_id basierend auf poster_type zuordnen
        """
        if not event_config.initial_posts:
            return event_config
        
        # Agent-Index nach Entitätstyp erstellen
        agents_by_type: Dict[str, List[AgentActivityConfig]] = {}
        for agent in agent_configs:
            etype = agent.entity_type.lower()
            if etype not in agents_by_type:
                agents_by_type[etype] = []
            agents_by_type[etype].append(agent)
        
        # Typen-Zuordnungstabelle (Behandlung verschiedener LLM-Ausgabeformate)
        type_aliases = {
            "official": ["official", "university", "governmentagency", "government"],
            "university": ["university", "official"],
            "mediaoutlet": ["mediaoutlet", "media"],
            "student": ["student", "person"],
            "professor": ["professor", "expert", "teacher"],
            "alumni": ["alumni", "person"],
            "organization": ["organization", "ngo", "company", "group"],
            "person": ["person", "student", "alumni"],
        }
        
        # Verwendete Agent-Indizes pro Typ aufzeichnen, um doppelte Verwendung desselben Agents zu vermeiden
        used_indices: Dict[str, int] = {}
        
        updated_posts = []
        for post in event_config.initial_posts:
            poster_type = post.get("poster_type", "").lower()
            content = post.get("content", "")
            
            # Versuche passenden Agent zu finden
            matched_agent_id = None
            
            # 1. Direkte Übereinstimmung
            if poster_type in agents_by_type:
                agents = agents_by_type[poster_type]
                idx = used_indices.get(poster_type, 0) % len(agents)
                matched_agent_id = agents[idx].agent_id
                used_indices[poster_type] = idx + 1
            else:
                # 2. Alias-Übereinstimmung verwenden
                for alias_key, aliases in type_aliases.items():
                    if poster_type in aliases or alias_key == poster_type:
                        for alias in aliases:
                            if alias in agents_by_type:
                                agents = agents_by_type[alias]
                                idx = used_indices.get(alias, 0) % len(agents)
                                matched_agent_id = agents[idx].agent_id
                                used_indices[alias] = idx + 1
                                break
                    if matched_agent_id is not None:
                        break
            
            # 3. Falls immer noch nicht gefunden, Agent mit höchstem Einfluss verwenden
            if matched_agent_id is None:
                logger.warning(f"Kein passender Agent für Typ '{poster_type}' gefunden, verwende Agent mit höchstem Einfluss")
                if agent_configs:
                    # Nach Einfluss sortieren, den mit höchstem Einfluss auswählen
                    sorted_agents = sorted(agent_configs, key=lambda a: a.influence_weight, reverse=True)
                    matched_agent_id = sorted_agents[0].agent_id
                else:
                    matched_agent_id = 0
            
            updated_posts.append({
                "content": content,
                "poster_type": post.get("poster_type", "Unknown"),
                "poster_agent_id": matched_agent_id
            })
            
            logger.info(f"Anfangsbeitrag zugewiesen: poster_type='{poster_type}' -> agent_id={matched_agent_id}")
        
        event_config.initial_posts = updated_posts
        return event_config
    
    def _generate_agent_configs_batch(
        self,
        context: str,
        entities: List[EntityNode],
        start_idx: int,
        simulation_requirement: str
    ) -> List[AgentActivityConfig]:
        """Agent-Konfigurationen in Batches generieren"""
        
        # Entitätsinformationen aufbauen (konfigurierte Zusammenfassungslänge verwenden)
        entity_list = []
        summary_len = self.AGENT_SUMMARY_LENGTH
        for i, e in enumerate(entities):
            entity_list.append({
                "agent_id": start_idx + i,
                "entity_name": e.name,
                "entity_type": e.get_entity_type() or "Unknown",
                "summary": e.summary[:summary_len] if e.summary else ""
            })
        
        prompt = f"""Basierend auf den folgenden Informationen, generiere für jede Entität eine Social-Media-Aktivitätskonfiguration.

Simulationsanforderungen: {simulation_requirement}

## Entitätenliste
```json
{json.dumps(entity_list, ensure_ascii=False, indent=2)}
```

## Aufgabe
Generiere eine Aktivitätskonfiguration für jede Entität. Beachte:
- **Zeiten entsprechend der Zielnutzergruppe**: Folgendes dient als Referenz (UTC+8), bitte je nach Simulationsszenario anpassen
- **Offizielle Institutionen** (University/GovernmentAgency): Niedrige Aktivität (0.1-0.3), Arbeitszeiten (9-17) aktiv, langsame Reaktion (60-240 Min.), hoher Einfluss (2.5-3.0)
- **Medien** (MediaOutlet): Mittlere Aktivität (0.4-0.6), ganztägig aktiv (8-23), schnelle Reaktion (5-30 Min.), hoher Einfluss (2.0-2.5)
- **Einzelpersonen** (Student/Person/Alumni): Hohe Aktivität (0.6-0.9), hauptsächlich abends aktiv (18-23), schnelle Reaktion (1-15 Min.), geringer Einfluss (0.8-1.2)
- **Öffentliche Persönlichkeiten/Experten**: Mittlere Aktivität (0.4-0.6), mittelhoher Einfluss (1.5-2.0)

JSON-Rückgabeformat (kein Markdown):
{{
    "agent_configs": [
        {{
            "agent_id": <muss mit Eingabe übereinstimmen>,
            "activity_level": <0.0-1.0>,
            "posts_per_hour": <Beitragshäufigkeit>,
            "comments_per_hour": <Kommentarhäufigkeit>,
            "active_hours": [<Liste aktiver Stunden, Tagesrhythmus der Zielgruppe berücksichtigen>],
            "response_delay_min": <Minimale Reaktionsverzögerung in Minuten>,
            "response_delay_max": <Maximale Reaktionsverzögerung in Minuten>,
            "sentiment_bias": <-1.0 bis 1.0>,
            "stance": "<supportive/opposing/neutral/observer>",
            "influence_weight": <Einflussgewichtung>
        }},
        ...
    ]
}}"""

        system_prompt = "Du bist ein Experte für Social-Media-Verhaltensanalyse. Gib reines JSON zurück. Die Konfiguration muss den Tagesrhythmen der Zielnutzergruppe im Simulationsszenario entsprechen."
        system_prompt = f"{system_prompt}\n\n{get_language_instruction()}\nIMPORTANT: The 'stance' field value MUST be one of the English strings: 'supportive', 'opposing', 'neutral', 'observer'. All JSON field names and numeric values must remain unchanged. Only natural language text fields should use the specified language."

        try:
            result = self._call_llm_with_retry(prompt, system_prompt)
            llm_configs = {cfg["agent_id"]: cfg for cfg in result.get("agent_configs", [])}
        except Exception as e:
            logger.warning(f"LLM-Generierung des Agent-Konfigurations-Batches fehlgeschlagen: {e}, verwende regelbasierte Generierung")
            llm_configs = {}
        
        # AgentActivityConfig-Objekte erstellen
        configs = []
        for i, entity in enumerate(entities):
            agent_id = start_idx + i
            cfg = llm_configs.get(agent_id, {})
            
            # Wenn LLM nicht generiert hat, regelbasierte Generierung verwenden
            if not cfg:
                cfg = self._generate_agent_config_by_rule(entity)
            
            config = AgentActivityConfig(
                agent_id=agent_id,
                entity_uuid=entity.uuid,
                entity_name=entity.name,
                entity_type=entity.get_entity_type() or "Unknown",
                activity_level=cfg.get("activity_level", 0.5),
                posts_per_hour=cfg.get("posts_per_hour", 0.5),
                comments_per_hour=cfg.get("comments_per_hour", 1.0),
                active_hours=cfg.get("active_hours", list(range(9, 23))),
                response_delay_min=cfg.get("response_delay_min", 5),
                response_delay_max=cfg.get("response_delay_max", 60),
                sentiment_bias=cfg.get("sentiment_bias", 0.0),
                stance=cfg.get("stance", "neutral"),
                influence_weight=cfg.get("influence_weight", 1.0)
            )
            configs.append(config)
        
        return configs
    
    def _generate_agent_config_by_rule(self, entity: EntityNode) -> Dict[str, Any]:
        """Regelbasierte Generierung einer einzelnen Agent-Konfiguration (chinesische Tagesrhythmen)"""
        entity_type = (entity.get_entity_type() or "Unknown").lower()
        
        if entity_type in ["university", "governmentagency", "ngo"]:
            # Offizielle Institutionen: Arbeitszeiten aktiv, niedrige Häufigkeit, hoher Einfluss
            return {
                "activity_level": 0.2,
                "posts_per_hour": 0.1,
                "comments_per_hour": 0.05,
                "active_hours": list(range(9, 18)),  # 9:00-17:59
                "response_delay_min": 60,
                "response_delay_max": 240,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 3.0
            }
        elif entity_type in ["mediaoutlet"]:
            # Medien: Ganztägig aktiv, mittlere Häufigkeit, hoher Einfluss
            return {
                "activity_level": 0.5,
                "posts_per_hour": 0.8,
                "comments_per_hour": 0.3,
                "active_hours": list(range(7, 24)),  # 7:00-23:59
                "response_delay_min": 5,
                "response_delay_max": 30,
                "sentiment_bias": 0.0,
                "stance": "observer",
                "influence_weight": 2.5
            }
        elif entity_type in ["professor", "expert", "official"]:
            # Experten/Professoren: Arbeitszeit + Abend aktiv, mittlere Häufigkeit
            return {
                "activity_level": 0.4,
                "posts_per_hour": 0.3,
                "comments_per_hour": 0.5,
                "active_hours": list(range(8, 22)),  # 8:00-21:59
                "response_delay_min": 15,
                "response_delay_max": 90,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 2.0
            }
        elif entity_type in ["student"]:
            # Studenten: Hauptsächlich abends, hohe Häufigkeit
            return {
                "activity_level": 0.8,
                "posts_per_hour": 0.6,
                "comments_per_hour": 1.5,
                "active_hours": [8, 9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23],  # Vormittags + Abends
                "response_delay_min": 1,
                "response_delay_max": 15,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 0.8
            }
        elif entity_type in ["alumni"]:
            # Alumni: Hauptsächlich abends
            return {
                "activity_level": 0.6,
                "posts_per_hour": 0.4,
                "comments_per_hour": 0.8,
                "active_hours": [12, 13, 19, 20, 21, 22, 23],  # Mittagspause + Abends
                "response_delay_min": 5,
                "response_delay_max": 30,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 1.0
            }
        else:
            # Normalpersonen: Abendliche Spitze
            return {
                "activity_level": 0.7,
                "posts_per_hour": 0.5,
                "comments_per_hour": 1.2,
                "active_hours": [9, 10, 11, 12, 13, 18, 19, 20, 21, 22, 23],  # Tagsüber + Abends
                "response_delay_min": 2,
                "response_delay_max": 20,
                "sentiment_bias": 0.0,
                "stance": "neutral",
                "influence_weight": 1.0
            }
    

