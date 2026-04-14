"""
Zep-Abrufdienst
Kapselt Graphsuche, Knotenlesen, Kantenabfragen und weitere Werkzeuge für den Report Agent.

Kern-Abrufwerkzeuge (optimiert):
1. InsightForge (Tiefenanalyse-Abruf) - Leistungsstärkstes hybrides Abrufwerkzeug, generiert automatisch Unterfragen und ruft mehrdimensional ab
2. PanoramaSearch (Breitensuche) - Gesamtüberblick einschließlich abgelaufener Inhalte
3. QuickSearch (Einfache Suche) - Schneller Abruf
"""

import time
import json
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field

from zep_cloud.client import Zep

from ..config import Config
from ..utils.logger import get_logger
from ..utils.llm_client import LLMClient
from ..utils.locale import get_locale, t
from ..utils.zep_paging import fetch_all_nodes, fetch_all_edges

logger = get_logger('mirofish.zep_tools')


@dataclass
class SearchResult:
    """Suchergebnis"""
    facts: List[str]
    edges: List[Dict[str, Any]]
    nodes: List[Dict[str, Any]]
    query: str
    total_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "facts": self.facts,
            "edges": self.edges,
            "nodes": self.nodes,
            "query": self.query,
            "total_count": self.total_count
        }
    
    def to_text(self) -> str:
        """In Textformat umwandeln, für LLM-Verständnis"""
        text_parts = [f"Suchabfrage: {self.query}", f"{self.total_count} relevante Informationen gefunden"]
        
        if self.facts:
            text_parts.append("\n### Relevante Fakten:")
            for i, fact in enumerate(self.facts, 1):
                text_parts.append(f"{i}. {fact}")
        
        return "\n".join(text_parts)


@dataclass
class NodeInfo:
    """Knoteninformation"""
    uuid: str
    name: str
    labels: List[str]
    summary: str
    attributes: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "labels": self.labels,
            "summary": self.summary,
            "attributes": self.attributes
        }
    
    def to_text(self) -> str:
        """In Textformat umwandeln"""
        entity_type = next((l for l in self.labels if l not in ["Entity", "Node"]), "Unbekannter Typ")
        return f"Entität: {self.name} (Typ: {entity_type})\nZusammenfassung: {self.summary}"


@dataclass
class EdgeInfo:
    """Kanteninformation"""
    uuid: str
    name: str
    fact: str
    source_node_uuid: str
    target_node_uuid: str
    source_node_name: Optional[str] = None
    target_node_name: Optional[str] = None
    # Zeitinformationen
    created_at: Optional[str] = None
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None
    expired_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "uuid": self.uuid,
            "name": self.name,
            "fact": self.fact,
            "source_node_uuid": self.source_node_uuid,
            "target_node_uuid": self.target_node_uuid,
            "source_node_name": self.source_node_name,
            "target_node_name": self.target_node_name,
            "created_at": self.created_at,
            "valid_at": self.valid_at,
            "invalid_at": self.invalid_at,
            "expired_at": self.expired_at
        }
    
    def to_text(self, include_temporal: bool = False) -> str:
        """In Textformat umwandeln"""
        source = self.source_node_name or self.source_node_uuid[:8]
        target = self.target_node_name or self.target_node_uuid[:8]
        base_text = f"Beziehung: {source} --[{self.name}]--> {target}\nFakt: {self.fact}"
        
        if include_temporal:
            valid_at = self.valid_at or "Unbekannt"
            invalid_at = self.invalid_at or "Bis heute"
            base_text += f"\nGültigkeitszeitraum: {valid_at} - {invalid_at}"
            if self.expired_at:
                base_text += f" (Abgelaufen: {self.expired_at})"
        
        return base_text
    
    @property
    def is_expired(self) -> bool:
        """Ob abgelaufen"""
        return self.expired_at is not None
    
    @property
    def is_invalid(self) -> bool:
        """Ob ungültig geworden"""
        return self.invalid_at is not None


@dataclass
class InsightForgeResult:
    """
    Tiefenanalyse-Abrufergebnis (InsightForge)
    Enthält Abrufergebnisse mehrerer Unterfragen sowie eine zusammenfassende Analyse
    """
    query: str
    simulation_requirement: str
    sub_queries: List[str]
    
    # Abrufergebnisse der verschiedenen Dimensionen
    semantic_facts: List[str] = field(default_factory=list)  # Semantische Suchergebnisse
    entity_insights: List[Dict[str, Any]] = field(default_factory=list)  # Entitäts-Erkenntnisse
    relationship_chains: List[str] = field(default_factory=list)  # Beziehungsketten
    
    # Statistikinformationen
    total_facts: int = 0
    total_entities: int = 0
    total_relationships: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "simulation_requirement": self.simulation_requirement,
            "sub_queries": self.sub_queries,
            "semantic_facts": self.semantic_facts,
            "entity_insights": self.entity_insights,
            "relationship_chains": self.relationship_chains,
            "total_facts": self.total_facts,
            "total_entities": self.total_entities,
            "total_relationships": self.total_relationships
        }
    
    def to_text(self) -> str:
        """In detailliertes Textformat umwandeln, für LLM-Verständnis"""
        text_parts = [
            f"## Zukunftsprognose Tiefenanalyse",
            f"Analysefrage: {self.query}",
            f"Prognoseszenario: {self.simulation_requirement}",
            f"\n### Prognosedaten-Statistik",
            f"- Relevante Prognosefakten: {self.total_facts} Einträge",
            f"- Beteiligte Entitäten: {self.total_entities} Stück",
            f"- Beziehungsketten: {self.total_relationships} Einträge"
        ]
        
        # Unterfragen
        if self.sub_queries:
            text_parts.append(f"\n### Analysierte Unterfragen")
            for i, sq in enumerate(self.sub_queries, 1):
                text_parts.append(f"{i}. {sq}")
        
        # Semantische Suchergebnisse
        if self.semantic_facts:
            text_parts.append(f"\n### 【Schlüsselfakten】(Bitte zitieren Sie diese Originaltexte im Bericht)")
            for i, fact in enumerate(self.semantic_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")
        
        # Entitäts-Erkenntnisse
        if self.entity_insights:
            text_parts.append(f"\n### 【Kernentitäten】")
            for entity in self.entity_insights:
                text_parts.append(f"- **{entity.get('name', 'Unbekannt')}** ({entity.get('type', 'Entität')})")
                if entity.get('summary'):
                    text_parts.append(f"  Zusammenfassung: \"{entity.get('summary')}\"")
                if entity.get('related_facts'):
                    text_parts.append(f"  Verwandte Fakten: {len(entity.get('related_facts', []))} Einträge")
        
        # Beziehungsketten
        if self.relationship_chains:
            text_parts.append(f"\n### 【Beziehungsketten】")
            for chain in self.relationship_chains:
                text_parts.append(f"- {chain}")
        
        return "\n".join(text_parts)


@dataclass
class PanoramaResult:
    """
    Breitensuchergebnis (Panorama)
    Enthält alle relevanten Informationen, einschließlich abgelaufener Inhalte
    """
    query: str
    
    # Alle Knoten
    all_nodes: List[NodeInfo] = field(default_factory=list)
    # Alle Kanten (einschließlich abgelaufener)
    all_edges: List[EdgeInfo] = field(default_factory=list)
    # Aktuell gültige Fakten
    active_facts: List[str] = field(default_factory=list)
    # Abgelaufene/ungültige Fakten (Verlauf)
    historical_facts: List[str] = field(default_factory=list)
    
    # Statistik
    total_nodes: int = 0
    total_edges: int = 0
    active_count: int = 0
    historical_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "all_nodes": [n.to_dict() for n in self.all_nodes],
            "all_edges": [e.to_dict() for e in self.all_edges],
            "active_facts": self.active_facts,
            "historical_facts": self.historical_facts,
            "total_nodes": self.total_nodes,
            "total_edges": self.total_edges,
            "active_count": self.active_count,
            "historical_count": self.historical_count
        }
    
    def to_text(self) -> str:
        """In Textformat umwandeln (Vollversion, nicht gekürzt)"""
        text_parts = [
            f"## Breitensuchergebnis (Zukunfts-Panoramaansicht)",
            f"Abfrage: {self.query}",
            f"\n### Statistikinformationen",
            f"- Gesamtanzahl Knoten: {self.total_nodes}",
            f"- Gesamtanzahl Kanten: {self.total_edges}",
            f"- Aktuell gültige Fakten: {self.active_count} Einträge",
            f"- Historische/abgelaufene Fakten: {self.historical_count} Einträge"
        ]
        
        # Aktuell gültige Fakten (vollständige Ausgabe, nicht gekürzt)
        if self.active_facts:
            text_parts.append(f"\n### 【Aktuell gültige Fakten】(Originaltext der Simulationsergebnisse)")
            for i, fact in enumerate(self.active_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")
        
        # Historische/abgelaufene Fakten (vollständige Ausgabe, nicht gekürzt)
        if self.historical_facts:
            text_parts.append(f"\n### 【Historische/abgelaufene Fakten】(Aufzeichnung des Entwicklungsprozesses)")
            for i, fact in enumerate(self.historical_facts, 1):
                text_parts.append(f"{i}. \"{fact}\"")
        
        # Schlüsselentitäten (vollständige Ausgabe, nicht gekürzt)
        if self.all_nodes:
            text_parts.append(f"\n### 【Beteiligte Entitäten】")
            for node in self.all_nodes:
                entity_type = next((l for l in node.labels if l not in ["Entity", "Node"]), "Entität")
                text_parts.append(f"- **{node.name}** ({entity_type})")
        
        return "\n".join(text_parts)


@dataclass
class AgentInterview:
    """Einzelnes Agent-Interviewergebnis"""
    agent_name: str
    agent_role: str  # Rollentyp (z.B.: Student, Lehrer, Medien usw.)
    agent_bio: str  # Kurzbiografie
    question: str  # Interviewfrage
    response: str  # Interviewantwort
    key_quotes: List[str] = field(default_factory=list)  # Schlüsselzitate
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "agent_role": self.agent_role,
            "agent_bio": self.agent_bio,
            "question": self.question,
            "response": self.response,
            "key_quotes": self.key_quotes
        }
    
    def to_text(self) -> str:
        text = f"**{self.agent_name}** ({self.agent_role})\n"
        # Vollständige agent_bio anzeigen, nicht kürzen
        text += f"_Kurzbiografie: {self.agent_bio}_\n\n"
        text += f"**Q:** {self.question}\n\n"
        text += f"**A:** {self.response}\n"
        if self.key_quotes:
            text += "\n**Schlüsselzitate:**\n"
            for quote in self.key_quotes:
                # Verschiedene Anführungszeichen bereinigen
                clean_quote = quote.replace('\u201c', '').replace('\u201d', '').replace('"', '')
                clean_quote = clean_quote.replace('\u300c', '').replace('\u300d', '')
                clean_quote = clean_quote.strip()
                # Führende Satzzeichen entfernen
                while clean_quote and clean_quote[0] in '，,；;：:、。！？\n\r\t ':
                    clean_quote = clean_quote[1:]
                # Störende Inhalte mit Fragennummern herausfiltern (Frage 1-9)
                skip = False
                for d in '123456789':
                    if f'\u95ee\u9898{d}' in clean_quote:
                        skip = True
                        break
                if skip:
                    continue
                # Zu lange Inhalte kürzen (am Satzzeichen trennen, nicht hart abschneiden)
                if len(clean_quote) > 150:
                    dot_pos = clean_quote.find('\u3002', 80)
                    if dot_pos > 0:
                        clean_quote = clean_quote[:dot_pos + 1]
                    else:
                        clean_quote = clean_quote[:147] + "..."
                if clean_quote and len(clean_quote) >= 10:
                    text += f'> "{clean_quote}"\n'
        return text


@dataclass
class InterviewResult:
    """
    Interviewergebnis (Interview)
    Enthält Interviewantworten mehrerer simulierter Agents
    """
    interview_topic: str  # Interviewthema
    interview_questions: List[str]  # Liste der Interviewfragen
    
    # Für das Interview ausgewählte Agents
    selected_agents: List[Dict[str, Any]] = field(default_factory=list)
    # Interviewantworten der einzelnen Agents
    interviews: List[AgentInterview] = field(default_factory=list)
    
    # Begründung der Agent-Auswahl
    selection_reasoning: str = ""
    # Zusammengefasste Interviewzusammenfassung
    summary: str = ""
    
    # Statistik
    total_agents: int = 0
    interviewed_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "interview_topic": self.interview_topic,
            "interview_questions": self.interview_questions,
            "selected_agents": self.selected_agents,
            "interviews": [i.to_dict() for i in self.interviews],
            "selection_reasoning": self.selection_reasoning,
            "summary": self.summary,
            "total_agents": self.total_agents,
            "interviewed_count": self.interviewed_count
        }
    
    def to_text(self) -> str:
        """In detailliertes Textformat umwandeln, für LLM-Verständnis und Berichtszitate"""
        text_parts = [
            "## Tiefeninterview-Bericht",
            f"**Interviewthema:** {self.interview_topic}",
            f"**Anzahl Interviewte:** {self.interviewed_count} / {self.total_agents} simulierte Agents",
            "\n### Begründung der Interviewpartnerauswahl",
            self.selection_reasoning or "(Automatische Auswahl)",
            "\n---",
            "\n### Interviewprotokoll",
        ]

        if self.interviews:
            for i, interview in enumerate(self.interviews, 1):
                text_parts.append(f"\n#### Interview #{i}: {interview.agent_name}")
                text_parts.append(interview.to_text())
                text_parts.append("\n---")
        else:
            text_parts.append("(Keine Interviewaufzeichnungen)\n\n---")

        text_parts.append("\n### Interviewzusammenfassung und Kernaussagen")
        text_parts.append(self.summary or "(Keine Zusammenfassung)")

        return "\n".join(text_parts)


class ZepToolsService:
    """
    Zep-Abrufwerkzeug-Service
    
    【Kern-Abrufwerkzeuge - optimiert】
    1. insight_forge - Tiefenanalyse-Abruf (leistungsstärkste, automatische Unterfragengenerierung, mehrdimensionaler Abruf)
    2. panorama_search - Breitensuche (Gesamtüberblick, einschließlich abgelaufener Inhalte)
    3. quick_search - Einfache Suche (schneller Abruf)
    4. interview_agents - Tiefeninterview (Interview simulierter Agents, Erhalt von Mehrperspektiven-Standpunkten)
    
    【Basiswerkzeuge】
    - search_graph - Graph-Semantiksuche
    - get_all_nodes - Alle Knoten des Graphen abrufen
    - get_all_edges - Alle Kanten des Graphen abrufen (mit Zeitinformationen)
    - get_node_detail - Knotendetails abrufen
    - get_node_edges - Knotenbezogene Kanten abrufen
    - get_entities_by_type - Entitäten nach Typ abrufen
    - get_entity_summary - Beziehungszusammenfassung einer Entität abrufen
    """
    
    # Wiederholungskonfiguration
    MAX_RETRIES = 3
    RETRY_DELAY = 2.0
    
    def __init__(self, api_key: Optional[str] = None, llm_client: Optional[LLMClient] = None):
        self.api_key = api_key or Config.ZEP_API_KEY
        if not self.api_key:
            raise ValueError("ZEP_API_KEY nicht konfiguriert")
        
        self.client = Zep(api_key=self.api_key)
        # LLM-Client für InsightForge-Unterfragengenerierung
        self._llm_client = llm_client
        logger.info(t("console.zepToolsInitialized"))
    
    @property
    def llm(self) -> LLMClient:
        """Verzögerte Initialisierung des LLM-Clients"""
        if self._llm_client is None:
            self._llm_client = LLMClient()
        return self._llm_client
    
    def _call_with_retry(self, func, operation_name: str, max_retries: int = None):
        """API-Aufruf mit Wiederholungsmechanismus"""
        max_retries = max_retries or self.MAX_RETRIES
        last_exception = None
        delay = self.RETRY_DELAY
        
        for attempt in range(max_retries):
            try:
                return func()
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    logger.warning(
                        t("console.zepRetryAttempt", operation=operation_name, attempt=attempt + 1, error=str(e)[:100], delay=f"{delay:.1f}")
                    )
                    time.sleep(delay)
                    delay *= 2
                else:
                    logger.error(t("console.zepAllRetriesFailed", operation=operation_name, retries=max_retries, error=str(e)))
        
        raise last_exception
    
    def search_graph(
        self, 
        graph_id: str, 
        query: str, 
        limit: int = 10,
        scope: str = "edges"
    ) -> SearchResult:
        """
        Graph-Semantiksuche
        
        Verwendet Hybridsuche (Semantik+BM25) um im Graphen nach relevanten Informationen zu suchen.
        Falls die Zep Cloud Search-API nicht verfügbar ist, wird auf lokales Schlüsselwort-Matching zurückgegriffen.
        
        Args:
            graph_id: Graph-ID (Standalone Graph)
            query: Suchabfrage
            limit: Anzahl der Ergebnisse
            scope: Suchbereich, "edges" oder "nodes"
            
        Returns:
            SearchResult: Suchergebnis
        """
        logger.info(t("console.graphSearch", graphId=graph_id, query=query[:50]))
        
        # Versuch, die Zep Cloud Search API zu verwenden
        try:
            search_results = self._call_with_retry(
                func=lambda: self.client.graph.search(
                    graph_id=graph_id,
                    query=query,
                    limit=limit,
                    scope=scope,
                    reranker="cross_encoder"
                ),
                operation_name=t("console.graphSearchOp", graphId=graph_id)
            )
            
            facts = []
            edges = []
            nodes = []
            
            # Kanten-Suchergebnisse parsen
            if hasattr(search_results, 'edges') and search_results.edges:
                for edge in search_results.edges:
                    if hasattr(edge, 'fact') and edge.fact:
                        facts.append(edge.fact)
                    edges.append({
                        "uuid": getattr(edge, 'uuid_', None) or getattr(edge, 'uuid', ''),
                        "name": getattr(edge, 'name', ''),
                        "fact": getattr(edge, 'fact', ''),
                        "source_node_uuid": getattr(edge, 'source_node_uuid', ''),
                        "target_node_uuid": getattr(edge, 'target_node_uuid', ''),
                    })
            
            # Knoten-Suchergebnisse parsen
            if hasattr(search_results, 'nodes') and search_results.nodes:
                for node in search_results.nodes:
                    nodes.append({
                        "uuid": getattr(node, 'uuid_', None) or getattr(node, 'uuid', ''),
                        "name": getattr(node, 'name', ''),
                        "labels": getattr(node, 'labels', []),
                        "summary": getattr(node, 'summary', ''),
                    })
                    # Knotenzusammenfassung zählt auch als Fakt
                    if hasattr(node, 'summary') and node.summary:
                        facts.append(f"[{node.name}]: {node.summary}")
            
            logger.info(t("console.searchComplete", count=len(facts)))
            
            return SearchResult(
                facts=facts,
                edges=edges,
                nodes=nodes,
                query=query,
                total_count=len(facts)
            )
            
        except Exception as e:
            logger.warning(t("console.zepSearchApiFallback", error=str(e)))
            # Fallback: Lokales Schlüsselwort-Matching verwenden
            return self._local_search(graph_id, query, limit, scope)
    
    def _local_search(
        self, 
        graph_id: str, 
        query: str, 
        limit: int = 10,
        scope: str = "edges"
    ) -> SearchResult:
        """
        Lokale Schlüsselwort-Matching-Suche (als Fallback für die Zep Search API)
        
        Ruft alle Kanten/Knoten ab und führt dann lokales Schlüsselwort-Matching durch
        
        Args:
            graph_id: Graph-ID
            query: Suchabfrage
            limit: Anzahl der Ergebnisse
            scope: Suchbereich
            
        Returns:
            SearchResult: Suchergebnis
        """
        logger.info(t("console.usingLocalSearch", query=query[:30]))
        
        facts = []
        edges_result = []
        nodes_result = []
        
        # Suchbegriffe extrahieren (einfache Tokenisierung)
        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace('，', ' ').split() if len(w.strip()) > 1]
        
        def match_score(text: str) -> int:
            """Übereinstimmungswert zwischen Text und Abfrage berechnen"""
            if not text:
                return 0
            text_lower = text.lower()
            # Vollständige Übereinstimmung mit der Abfrage
            if query_lower in text_lower:
                return 100
            # Schlüsselwort-Übereinstimmung
            score = 0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 10
            return score
        
        try:
            if scope in ["edges", "both"]:
                # Alle Kanten abrufen und abgleichen
                all_edges = self.get_all_edges(graph_id)
                scored_edges = []
                for edge in all_edges:
                    score = match_score(edge.fact) + match_score(edge.name)
                    if score > 0:
                        scored_edges.append((score, edge))
                
                # Nach Punktzahl sortieren
                scored_edges.sort(key=lambda x: x[0], reverse=True)
                
                for score, edge in scored_edges[:limit]:
                    if edge.fact:
                        facts.append(edge.fact)
                    edges_result.append({
                        "uuid": edge.uuid,
                        "name": edge.name,
                        "fact": edge.fact,
                        "source_node_uuid": edge.source_node_uuid,
                        "target_node_uuid": edge.target_node_uuid,
                    })
            
            if scope in ["nodes", "both"]:
                # Alle Knoten abrufen und abgleichen
                all_nodes = self.get_all_nodes(graph_id)
                scored_nodes = []
                for node in all_nodes:
                    score = match_score(node.name) + match_score(node.summary)
                    if score > 0:
                        scored_nodes.append((score, node))
                
                scored_nodes.sort(key=lambda x: x[0], reverse=True)
                
                for score, node in scored_nodes[:limit]:
                    nodes_result.append({
                        "uuid": node.uuid,
                        "name": node.name,
                        "labels": node.labels,
                        "summary": node.summary,
                    })
                    if node.summary:
                        facts.append(f"[{node.name}]: {node.summary}")
            
            logger.info(t("console.localSearchComplete", count=len(facts)))
            
        except Exception as e:
            logger.error(t("console.localSearchFailed", error=str(e)))
        
        return SearchResult(
            facts=facts,
            edges=edges_result,
            nodes=nodes_result,
            query=query,
            total_count=len(facts)
        )
    
    def get_all_nodes(self, graph_id: str) -> List[NodeInfo]:
        """
        Alle Knoten des Graphen abrufen (paginiert)

        Args:
            graph_id: Graph-ID

        Returns:
            Knotenliste
        """
        logger.info(t("console.fetchingAllNodes", graphId=graph_id))

        nodes = fetch_all_nodes(self.client, graph_id)

        result = []
        for node in nodes:
            node_uuid = getattr(node, 'uuid_', None) or getattr(node, 'uuid', None) or ""
            result.append(NodeInfo(
                uuid=str(node_uuid) if node_uuid else "",
                name=node.name or "",
                labels=node.labels or [],
                summary=node.summary or "",
                attributes=node.attributes or {}
            ))

        logger.info(t("console.fetchedNodes", count=len(result)))
        return result

    def get_all_edges(self, graph_id: str, include_temporal: bool = True) -> List[EdgeInfo]:
        """
        Alle Kanten des Graphen abrufen (paginiert, mit Zeitinformationen)

        Args:
            graph_id: Graph-ID
            include_temporal: Ob Zeitinformationen einbezogen werden sollen (Standard True)

        Returns:
            Kantenliste (enthält created_at, valid_at, invalid_at, expired_at)
        """
        logger.info(t("console.fetchingAllEdges", graphId=graph_id))

        edges = fetch_all_edges(self.client, graph_id)

        result = []
        for edge in edges:
            edge_uuid = getattr(edge, 'uuid_', None) or getattr(edge, 'uuid', None) or ""
            edge_info = EdgeInfo(
                uuid=str(edge_uuid) if edge_uuid else "",
                name=edge.name or "",
                fact=edge.fact or "",
                source_node_uuid=edge.source_node_uuid or "",
                target_node_uuid=edge.target_node_uuid or ""
            )

            # Zeitinformationen hinzufügen
            if include_temporal:
                edge_info.created_at = getattr(edge, 'created_at', None)
                edge_info.valid_at = getattr(edge, 'valid_at', None)
                edge_info.invalid_at = getattr(edge, 'invalid_at', None)
                edge_info.expired_at = getattr(edge, 'expired_at', None)

            result.append(edge_info)

        logger.info(t("console.fetchedEdges", count=len(result)))
        return result
    
    def get_node_detail(self, node_uuid: str) -> Optional[NodeInfo]:
        """
        Detailinformationen eines einzelnen Knotens abrufen
        
        Args:
            node_uuid: Knoten-UUID
            
        Returns:
            Knoteninformation oder None
        """
        logger.info(t("console.fetchingNodeDetail", uuid=node_uuid[:8]))
        
        try:
            node = self._call_with_retry(
                func=lambda: self.client.graph.node.get(uuid_=node_uuid),
                operation_name=t("console.fetchNodeDetailOp", uuid=node_uuid[:8])
            )
            
            if not node:
                return None
            
            return NodeInfo(
                uuid=getattr(node, 'uuid_', None) or getattr(node, 'uuid', ''),
                name=node.name or "",
                labels=node.labels or [],
                summary=node.summary or "",
                attributes=node.attributes or {}
            )
        except Exception as e:
            logger.error(t("console.fetchNodeDetailFailed", error=str(e)))
            return None
    
    def get_node_edges(self, graph_id: str, node_uuid: str) -> List[EdgeInfo]:
        """
        Alle Kanten eines Knotens abrufen
        
        Ruft alle Kanten des Graphen ab und filtert dann die mit dem angegebenen Knoten verbundenen Kanten
        
        Args:
            graph_id: Graph-ID
            node_uuid: Knoten-UUID
            
        Returns:
            Kantenliste
        """
        logger.info(t("console.fetchingNodeEdges", uuid=node_uuid[:8]))
        
        try:
            # Alle Kanten des Graphen abrufen, dann filtern
            all_edges = self.get_all_edges(graph_id)
            
            result = []
            for edge in all_edges:
                # Prüfen, ob die Kante mit dem angegebenen Knoten verbunden ist (als Quelle oder Ziel)
                if edge.source_node_uuid == node_uuid or edge.target_node_uuid == node_uuid:
                    result.append(edge)
            
            logger.info(t("console.foundNodeEdges", count=len(result)))
            return result
            
        except Exception as e:
            logger.warning(t("console.fetchNodeEdgesFailed", error=str(e)))
            return []
    
    def get_entities_by_type(
        self, 
        graph_id: str, 
        entity_type: str
    ) -> List[NodeInfo]:
        """
        Entitäten nach Typ abrufen
        
        Args:
            graph_id: Graph-ID
            entity_type: Entitätstyp (z.B. Student, PublicFigure usw.)
            
        Returns:
            Liste der Entitäten des angegebenen Typs
        """
        logger.info(t("console.fetchingEntitiesByType", type=entity_type))
        
        all_nodes = self.get_all_nodes(graph_id)
        
        filtered = []
        for node in all_nodes:
            # Prüfen, ob Labels den angegebenen Typ enthalten
            if entity_type in node.labels:
                filtered.append(node)
        
        logger.info(t("console.foundEntitiesByType", count=len(filtered), type=entity_type))
        return filtered
    
    def get_entity_summary(
        self, 
        graph_id: str, 
        entity_name: str
    ) -> Dict[str, Any]:
        """
        Beziehungszusammenfassung einer angegebenen Entität abrufen
        
        Sucht alle mit dieser Entität verbundenen Informationen und erstellt eine Zusammenfassung
        
        Args:
            graph_id: Graph-ID
            entity_name: Entitätsname
            
        Returns:
            Zusammenfassungsinformationen der Entität
        """
        logger.info(t("console.fetchingEntitySummary", name=entity_name))
        
        # Zuerst nach Informationen zu dieser Entität suchen
        search_result = self.search_graph(
            graph_id=graph_id,
            query=entity_name,
            limit=20
        )
        
        # Versuchen, die Entität in allen Knoten zu finden
        all_nodes = self.get_all_nodes(graph_id)
        entity_node = None
        for node in all_nodes:
            if node.name.lower() == entity_name.lower():
                entity_node = node
                break
        
        related_edges = []
        if entity_node:
            # graph_id-Parameter übergeben
            related_edges = self.get_node_edges(graph_id, entity_node.uuid)
        
        return {
            "entity_name": entity_name,
            "entity_info": entity_node.to_dict() if entity_node else None,
            "related_facts": search_result.facts,
            "related_edges": [e.to_dict() for e in related_edges],
            "total_relations": len(related_edges)
        }
    
    def get_graph_statistics(self, graph_id: str) -> Dict[str, Any]:
        """
        Statistikinformationen des Graphen abrufen
        
        Args:
            graph_id: Graph-ID
            
        Returns:
            Statistikinformationen
        """
        logger.info(t("console.fetchingGraphStats", graphId=graph_id))
        
        nodes = self.get_all_nodes(graph_id)
        edges = self.get_all_edges(graph_id)
        
        # Entitätstyp-Verteilung statistisch erfassen
        entity_types = {}
        for node in nodes:
            for label in node.labels:
                if label not in ["Entity", "Node"]:
                    entity_types[label] = entity_types.get(label, 0) + 1
        
        # Beziehungstyp-Verteilung statistisch erfassen
        relation_types = {}
        for edge in edges:
            relation_types[edge.name] = relation_types.get(edge.name, 0) + 1
        
        return {
            "graph_id": graph_id,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "entity_types": entity_types,
            "relation_types": relation_types
        }
    
    def get_simulation_context(
        self, 
        graph_id: str,
        simulation_requirement: str,
        limit: int = 30
    ) -> Dict[str, Any]:
        """
        Simulationsbezogene Kontextinformationen abrufen
        
        Umfassende Suche nach allen mit den Simulationsanforderungen verbundenen Informationen
        
        Args:
            graph_id: Graph-ID
            simulation_requirement: Beschreibung der Simulationsanforderung
            limit: Mengenbegrenzung pro Informationstyp
            
        Returns:
            Simulationskontextinformationen
        """
        logger.info(t("console.fetchingSimContext", requirement=simulation_requirement[:50]))
        
        # Nach simulationsrelevanten Informationen suchen
        search_result = self.search_graph(
            graph_id=graph_id,
            query=simulation_requirement,
            limit=limit
        )
        
        # Graph-Statistik abrufen
        stats = self.get_graph_statistics(graph_id)
        
        # Alle Entitätsknoten abrufen
        all_nodes = self.get_all_nodes(graph_id)
        
        # Entitäten mit tatsächlichem Typ filtern (keine reinen Entity-Knoten)
        entities = []
        for node in all_nodes:
            custom_labels = [l for l in node.labels if l not in ["Entity", "Node"]]
            if custom_labels:
                entities.append({
                    "name": node.name,
                    "type": custom_labels[0],
                    "summary": node.summary
                })
        
        return {
            "simulation_requirement": simulation_requirement,
            "related_facts": search_result.facts,
            "graph_statistics": stats,
            "entities": entities[:limit],  # Menge begrenzen
            "total_entities": len(entities)
        }
    
    # ========== Kern-Abrufwerkzeuge (optimiert) ==========
    
    def insight_forge(
        self,
        graph_id: str,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_sub_queries: int = 5
    ) -> InsightForgeResult:
        """
        【InsightForge - Tiefenanalyse-Abruf】
        
        Leistungsstärkste hybride Abruffunktion, zerlegt Fragen automatisch und ruft mehrdimensional ab:
        1. Verwendet LLM um die Frage in mehrere Unterfragen zu zerlegen
        2. Führt semantische Suche für jede Unterfrage durch
        3. Extrahiert relevante Entitäten und ruft deren Detailinformationen ab
        4. Verfolgt Beziehungsketten
        5. Integriert alle Ergebnisse und generiert tiefe Erkenntnisse
        
        Args:
            graph_id: Graph-ID
            query: Benutzerfrage
            simulation_requirement: Beschreibung der Simulationsanforderung
            report_context: Berichtskontext (optional, für präzisere Unterfragengenerierung)
            max_sub_queries: Maximale Anzahl der Unterfragen
            
        Returns:
            InsightForgeResult: Tiefenanalyse-Abrufergebnis
        """
        logger.info(t("console.insightForgeStart", query=query[:50]))
        
        result = InsightForgeResult(
            query=query,
            simulation_requirement=simulation_requirement,
            sub_queries=[]
        )
        
        # Step 1: Unterfragen mit LLM generieren
        sub_queries = self._generate_sub_queries(
            query=query,
            simulation_requirement=simulation_requirement,
            report_context=report_context,
            max_queries=max_sub_queries
        )
        result.sub_queries = sub_queries
        logger.info(t("console.generatedSubQueries", count=len(sub_queries)))
        
        # Step 2: Semantische Suche für jede Unterfrage
        all_facts = []
        all_edges = []
        seen_facts = set()
        
        for sub_query in sub_queries:
            search_result = self.search_graph(
                graph_id=graph_id,
                query=sub_query,
                limit=15,
                scope="edges"
            )
            
            for fact in search_result.facts:
                if fact not in seen_facts:
                    all_facts.append(fact)
                    seen_facts.add(fact)
            
            all_edges.extend(search_result.edges)
        
        # Auch die ursprüngliche Frage suchen
        main_search = self.search_graph(
            graph_id=graph_id,
            query=query,
            limit=20,
            scope="edges"
        )
        for fact in main_search.facts:
            if fact not in seen_facts:
                all_facts.append(fact)
                seen_facts.add(fact)
        
        result.semantic_facts = all_facts
        result.total_facts = len(all_facts)
        
        # Step 3: Relevante Entitäts-UUIDs aus Kanten extrahieren, nur deren Informationen abrufen (nicht alle Knoten)
        entity_uuids = set()
        for edge_data in all_edges:
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                if source_uuid:
                    entity_uuids.add(source_uuid)
                if target_uuid:
                    entity_uuids.add(target_uuid)
        
        # Details aller relevanten Entitäten abrufen (unbegrenzt, vollständige Ausgabe)
        entity_insights = []
        node_map = {}  # Für spätere Beziehungskettenbildung
        
        for uuid in list(entity_uuids):  # Alle Entitäten verarbeiten, nicht kürzen
            if not uuid:
                continue
            try:
                # Informationen jedes relevanten Knotens einzeln abrufen
                node = self.get_node_detail(uuid)
                if node:
                    node_map[uuid] = node
                    entity_type = next((l for l in node.labels if l not in ["Entity", "Node"]), "Entität")
                    
                    # Alle Fakten zu dieser Entität abrufen (nicht kürzen)
                    related_facts = [
                        f for f in all_facts 
                        if node.name.lower() in f.lower()
                    ]
                    
                    entity_insights.append({
                        "uuid": node.uuid,
                        "name": node.name,
                        "type": entity_type,
                        "summary": node.summary,
                        "related_facts": related_facts  # Vollständige Ausgabe, nicht kürzen
                    })
            except Exception as e:
                logger.debug(f"Knoten {uuid} abrufen fehlgeschlagen: {e}")
                continue
        
        result.entity_insights = entity_insights
        result.total_entities = len(entity_insights)
        
        # Step 4: Alle Beziehungsketten aufbauen (unbegrenzt)
        relationship_chains = []
        for edge_data in all_edges:  # Alle Kanten verarbeiten, nicht kürzen
            if isinstance(edge_data, dict):
                source_uuid = edge_data.get('source_node_uuid', '')
                target_uuid = edge_data.get('target_node_uuid', '')
                relation_name = edge_data.get('name', '')
                
                source_name = node_map.get(source_uuid, NodeInfo('', '', [], '', {})).name or source_uuid[:8]
                target_name = node_map.get(target_uuid, NodeInfo('', '', [], '', {})).name or target_uuid[:8]
                
                chain = f"{source_name} --[{relation_name}]--> {target_name}"
                if chain not in relationship_chains:
                    relationship_chains.append(chain)
        
        result.relationship_chains = relationship_chains
        result.total_relationships = len(relationship_chains)
        
        logger.info(t("console.insightForgeComplete", facts=result.total_facts, entities=result.total_entities, relationships=result.total_relationships))
        return result
    
    def _generate_sub_queries(
        self,
        query: str,
        simulation_requirement: str,
        report_context: str = "",
        max_queries: int = 5
    ) -> List[str]:
        """
        Unterfragen mit LLM generieren
        
        Zerlegt eine komplexe Frage in mehrere unabhängig abrufbare Unterfragen
        """
        system_prompt = """Du bist ein professioneller Frageanalyse-Experte. Deine Aufgabe ist es, eine komplexe Frage in mehrere Unterfragen zu zerlegen, die in einer simulierten Welt unabhängig beobachtet werden können.

Anforderungen:
1. Jede Unterfrage sollte spezifisch genug sein, um in der simulierten Welt relevantes Agent-Verhalten oder Ereignisse zu finden
2. Die Unterfragen sollten verschiedene Dimensionen der ursprünglichen Frage abdecken (z.B.: Wer, Was, Warum, Wie, Wann, Wo)
3. Die Unterfragen sollten mit dem Simulationsszenario zusammenhängen
4. Rückgabe im JSON-Format: {"sub_queries": ["Unterfrage1", "Unterfrage2", ...]}"""

        user_prompt = f"""Simulationsanforderung Hintergrund:
{simulation_requirement}

{f"Berichtskontext: {report_context[:500]}" if report_context else ""}

Bitte zerlegen Sie die folgende Frage in {max_queries} Unterfragen:
{query}

Geben Sie die Unterfragenliste im JSON-Format zurück."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            sub_queries = response.get("sub_queries", [])
            # Sicherstellen, dass es eine Liste von Strings ist
            return [str(sq) for sq in sub_queries[:max_queries]]
            
        except Exception as e:
            logger.warning(t("console.generateSubQueriesFailed", error=str(e)))
            # Fallback: Varianten basierend auf der ursprünglichen Frage zurückgeben
            return [
                query,
                f"Hauptbeteiligte von {query}",
                f"Ursachen und Auswirkungen von {query}",
                f"Entwicklungsprozess von {query}"
            ][:max_queries]
    
    def panorama_search(
        self,
        graph_id: str,
        query: str,
        include_expired: bool = True,
        limit: int = 50
    ) -> PanoramaResult:
        """
        【PanoramaSearch - Breitensuche】
        
        Gesamtüberblick abrufen, einschließlich aller relevanten Inhalte und historischer/abgelaufener Informationen:
        1. Alle relevanten Knoten abrufen
        2. Alle Kanten abrufen (einschließlich abgelaufener/ungültiger)
        3. Aktuell gültige und historische Informationen kategorisch sortieren
        
        Dieses Werkzeug eignet sich für Szenarien, in denen ein vollständiges Bild der Ereignisse benötigt wird oder der Entwicklungsprozess verfolgt werden soll.
        
        Args:
            graph_id: Graph-ID
            query: Suchabfrage (für Relevanz-Sortierung)
            include_expired: Ob abgelaufene Inhalte einbezogen werden sollen (Standard True)
            limit: Begrenzung der Ergebnisanzahl
            
        Returns:
            PanoramaResult: Breitensuchergebnis
        """
        logger.info(t("console.panoramaSearchStart", query=query[:50]))
        
        result = PanoramaResult(query=query)
        
        # Alle Knoten abrufen
        all_nodes = self.get_all_nodes(graph_id)
        node_map = {n.uuid: n for n in all_nodes}
        result.all_nodes = all_nodes
        result.total_nodes = len(all_nodes)
        
        # Alle Kanten abrufen (mit Zeitinformationen)
        all_edges = self.get_all_edges(graph_id, include_temporal=True)
        result.all_edges = all_edges
        result.total_edges = len(all_edges)
        
        # Fakten kategorisieren
        active_facts = []
        historical_facts = []
        
        for edge in all_edges:
            if not edge.fact:
                continue
            
            # Entitätsnamen zu Fakten hinzufügen
            source_name = node_map.get(edge.source_node_uuid, NodeInfo('', '', [], '', {})).name or edge.source_node_uuid[:8]
            target_name = node_map.get(edge.target_node_uuid, NodeInfo('', '', [], '', {})).name or edge.target_node_uuid[:8]
            
            # Prüfen, ob abgelaufen/ungültig
            is_historical = edge.is_expired or edge.is_invalid
            
            if is_historical:
                # Historische/abgelaufene Fakten, Zeitmarkierung hinzufügen
                valid_at = edge.valid_at or "Unbekannt"
                invalid_at = edge.invalid_at or edge.expired_at or "Unbekannt"
                fact_with_time = f"[{valid_at} - {invalid_at}] {edge.fact}"
                historical_facts.append(fact_with_time)
            else:
                # Aktuell gültige Fakten
                active_facts.append(edge.fact)
        
        # Relevanzbasierte Sortierung anhand der Abfrage
        query_lower = query.lower()
        keywords = [w.strip() for w in query_lower.replace(',', ' ').replace('，', ' ').split() if len(w.strip()) > 1]
        
        def relevance_score(fact: str) -> int:
            fact_lower = fact.lower()
            score = 0
            if query_lower in fact_lower:
                score += 100
            for kw in keywords:
                if kw in fact_lower:
                    score += 10
            return score
        
        # Sortieren und Menge begrenzen
        active_facts.sort(key=relevance_score, reverse=True)
        historical_facts.sort(key=relevance_score, reverse=True)
        
        result.active_facts = active_facts[:limit]
        result.historical_facts = historical_facts[:limit] if include_expired else []
        result.active_count = len(active_facts)
        result.historical_count = len(historical_facts)
        
        logger.info(t("console.panoramaSearchComplete", active=result.active_count, historical=result.historical_count))
        return result
    
    def quick_search(
        self,
        graph_id: str,
        query: str,
        limit: int = 10
    ) -> SearchResult:
        """
        【QuickSearch - Einfache Suche】
        
        Schnelles, leichtgewichtiges Abrufwerkzeug:
        1. Ruft direkt die Zep-Semantiksuche auf
        2. Gibt die relevantesten Ergebnisse zurück
        3. Geeignet für einfache, direkte Abrufanforderungen
        
        Args:
            graph_id: Graph-ID
            query: Suchabfrage
            limit: Anzahl der Ergebnisse
            
        Returns:
            SearchResult: Suchergebnis
        """
        logger.info(t("console.quickSearchStart", query=query[:50]))
        
        # Vorhandene search_graph-Methode direkt aufrufen
        result = self.search_graph(
            graph_id=graph_id,
            query=query,
            limit=limit,
            scope="edges"
        )
        
        logger.info(t("console.quickSearchComplete", count=result.total_count))
        return result
    
    def interview_agents(
        self,
        simulation_id: str,
        interview_requirement: str,
        simulation_requirement: str = "",
        max_agents: int = 5,
        custom_questions: List[str] = None
    ) -> InterviewResult:
        """
        【InterviewAgents - Tiefeninterview】
        
        Ruft die echte OASIS-Interview-API auf, um laufende Agents in der Simulation zu interviewen:
        1. Liest automatisch die Persona-Datei, um alle Simulations-Agents zu verstehen
        2. Verwendet LLM zur Analyse der Interviewanforderungen, wählt intelligent die relevantesten Agents aus
        3. Verwendet LLM zur Generierung von Interviewfragen
        4. Ruft /api/simulation/interview/batch-Schnittstelle für echte Interviews auf (beide Plattformen gleichzeitig)
        5. Integriert alle Interviewergebnisse und erstellt einen Interviewbericht
        
        【WICHTIG】Diese Funktion erfordert, dass die Simulationsumgebung läuft (OASIS-Umgebung nicht geschlossen)
        
        【Anwendungsszenarien】
        - Verständnis von Ereignissen aus verschiedenen Rollenperspektiven
        - Sammlung von Meinungen und Standpunkten verschiedener Parteien
        - Echte Antworten von Simulations-Agents erhalten (nicht LLM-simuliert)
        
        Args:
            simulation_id: Simulations-ID (zum Auffinden der Persona-Datei und Aufruf der Interview-API)
            interview_requirement: Beschreibung der Interviewanforderung (unstrukturiert, z.B. "Meinungen der Studenten zum Ereignis verstehen")
            simulation_requirement: Hintergrund der Simulationsanforderung (optional)
            max_agents: Maximale Anzahl zu interviewender Agents
            custom_questions: Benutzerdefinierte Interviewfragen (optional, werden automatisch generiert wenn nicht angegeben)
            
        Returns:
            InterviewResult: Interviewergebnis
        """
        from .simulation_runner import SimulationRunner
        
        logger.info(t("console.interviewAgentsStart", requirement=interview_requirement[:50]))
        
        result = InterviewResult(
            interview_topic=interview_requirement,
            interview_questions=custom_questions or []
        )
        
        # Step 1: Persona-Datei lesen
        profiles = self._load_agent_profiles(simulation_id)
        
        if not profiles:
            logger.warning(t("console.profilesNotFound", simId=simulation_id))
            result.summary = "Keine interviewbaren Agent-Persona-Dateien gefunden"
            return result
        
        result.total_agents = len(profiles)
        logger.info(t("console.loadedProfiles", count=len(profiles)))
        
        # Step 2: LLM verwenden um zu interviewende Agents auszuwählen (gibt agent_id-Liste zurück)
        selected_agents, selected_indices, selection_reasoning = self._select_agents_for_interview(
            profiles=profiles,
            interview_requirement=interview_requirement,
            simulation_requirement=simulation_requirement,
            max_agents=max_agents
        )
        
        result.selected_agents = selected_agents
        result.selection_reasoning = selection_reasoning
        logger.info(t("console.selectedAgentsForInterview", count=len(selected_agents), indices=selected_indices))
        
        # Step 3: Interviewfragen generieren (falls nicht bereitgestellt)
        if not result.interview_questions:
            result.interview_questions = self._generate_interview_questions(
                interview_requirement=interview_requirement,
                simulation_requirement=simulation_requirement,
                selected_agents=selected_agents
            )
            logger.info(t("console.generatedInterviewQuestions", count=len(result.interview_questions)))
        
        # Fragen zu einem Interview-Prompt zusammenführen
        combined_prompt = "\n".join([f"{i+1}. {q}" for i, q in enumerate(result.interview_questions)])
        
        # Optimierungspräfix hinzufügen, um das Agent-Antwortformat einzuschränken
        INTERVIEW_PROMPT_PREFIX = (
            "Du wirst gerade interviewt. Bitte beantworte die folgenden Fragen basierend auf deiner Persona, "
            "allen bisherigen Erinnerungen und Handlungen in reinem Textformat.\n"
            "Antwortanforderungen:\n"
            "1. Antworte direkt in natürlicher Sprache, rufe keine Werkzeuge auf\n"
            "2. Gib kein JSON-Format oder Werkzeugaufruf-Format zurück\n"
            "3. Verwende keine Markdown-Überschriften (wie #, ##, ###)\n"
            "4. Beantworte jede Frage der Reihe nach, beginne jede Antwort mit 'Frage X:' (X ist die Fragennummer)\n"
            "5. Trenne die Antworten auf verschiedene Fragen durch Leerzeilen\n"
            "6. Antworten sollten inhaltlich substanziell sein, mindestens 2-3 Sätze pro Frage\n\n"
        )
        optimized_prompt = f"{INTERVIEW_PROMPT_PREFIX}{combined_prompt}"
        
        # Step 4: Echte Interview-API aufrufen (ohne platform-Angabe, standardmäßig beide Plattformen gleichzeitig)
        try:
            # Batch-Interview-Liste erstellen (ohne platform-Angabe, beide Plattformen)
            interviews_request = []
            for agent_idx in selected_indices:
                interviews_request.append({
                    "agent_id": agent_idx,
                    "prompt": optimized_prompt  # Optimierten Prompt verwenden
                    # Ohne platform-Angabe, API interviewt auf Twitter und Reddit
                })
            
            logger.info(t("console.callingBatchInterviewApi", count=len(interviews_request)))
            
            # Batch-Interview-Methode des SimulationRunners aufrufen (ohne platform, beide Plattformen)
            api_result = SimulationRunner.interview_agents_batch(
                simulation_id=simulation_id,
                interviews=interviews_request,
                platform=None,  # Ohne platform-Angabe, beide Plattformen
                timeout=180.0   # Beide Plattformen benötigen längeres Timeout
            )
            
            logger.info(t("console.interviewApiReturned", count=api_result.get('interviews_count', 0), success=api_result.get('success')))
            
            # Prüfen, ob API-Aufruf erfolgreich war
            if not api_result.get("success", False):
                error_msg = api_result.get("error", "Unbekannter Fehler")
                logger.warning(t("console.interviewApiReturnedFailure", error=error_msg))
                result.summary = f"Interview-API-Aufruf fehlgeschlagen: {error_msg}. Bitte prüfen Sie den Status der OASIS-Simulationsumgebung."
                return result
            
            # Step 5: API-Rückgabeergebnisse parsen, AgentInterview-Objekte erstellen
            # Dual-Plattform-Modus Rückgabeformat: {"twitter_0": {...}, "reddit_0": {...}, "twitter_1": {...}, ...}
            api_data = api_result.get("result", {})
            results_dict = api_data.get("results", {}) if isinstance(api_data, dict) else {}
            
            for i, agent_idx in enumerate(selected_indices):
                agent = selected_agents[i]
                agent_name = agent.get("realname", agent.get("username", f"Agent_{agent_idx}"))
                agent_role = agent.get("profession", "Unbekannt")
                agent_bio = agent.get("bio", "")
                
                # Interviewergebnisse dieses Agents auf beiden Plattformen abrufen
                twitter_result = results_dict.get(f"twitter_{agent_idx}", {})
                reddit_result = results_dict.get(f"reddit_{agent_idx}", {})
                
                twitter_response = twitter_result.get("response", "")
                reddit_response = reddit_result.get("response", "")

                # Mögliche JSON-Werkzeugaufruf-Verpackung bereinigen
                twitter_response = self._clean_tool_call_response(twitter_response)
                reddit_response = self._clean_tool_call_response(reddit_response)

                # Immer Dual-Plattform-Markierungen ausgeben
                twitter_text = twitter_response if twitter_response else "(Keine Antwort von dieser Plattform)"
                reddit_text = reddit_response if reddit_response else "(Keine Antwort von dieser Plattform)"
                response_text = f"【Twitter-Plattform Antwort】\n{twitter_text}\n\n【Reddit-Plattform Antwort】\n{reddit_text}"

                # Schlüsselzitate extrahieren (aus den Antworten beider Plattformen)
                import re
                combined_responses = f"{twitter_response} {reddit_response}"

                # Antworttext bereinigen: Markierungen, Nummerierungen, Markdown usw. entfernen
                clean_text = re.sub(r'#{1,6}\s+', '', combined_responses)
                clean_text = re.sub(r'\{[^}]*tool_name[^}]*\}', '', clean_text)
                clean_text = re.sub(r'[*_`|>~\-]{2,}', '', clean_text)
                clean_text = re.sub(r'问题\d+[：:]\s*', '', clean_text)
                clean_text = re.sub(r'【[^】]+】', '', clean_text)

                # Strategie 1 (Hauptstrategie): Vollständige Sätze mit substanziellem Inhalt extrahieren
                sentences = re.split(r'[。！？]', clean_text)
                meaningful = [
                    s.strip() for s in sentences
                    if 20 <= len(s.strip()) <= 150
                    and not re.match(r'^[\s\W，,；;：:、]+', s.strip())
                    and not s.strip().startswith(('{', '问题'))
                ]
                meaningful.sort(key=len, reverse=True)
                key_quotes = [s + "。" for s in meaningful[:3]]

                # Strategie 2 (Ergänzung): Längerer Text in korrekt gepaarten chinesischen Anführungszeichen「」
                if not key_quotes:
                    paired = re.findall(r'\u201c([^\u201c\u201d]{15,100})\u201d', clean_text)
                    paired += re.findall(r'\u300c([^\u300c\u300d]{15,100})\u300d', clean_text)
                    key_quotes = [q for q in paired if not re.match(r'^[，,；;：:、]', q)][:3]
                
                interview = AgentInterview(
                    agent_name=agent_name,
                    agent_role=agent_role,
                    agent_bio=agent_bio[:1000],  # Bio-Längenbegrenzung erweitern
                    question=combined_prompt,
                    response=response_text,
                    key_quotes=key_quotes[:5]
                )
                result.interviews.append(interview)
            
            result.interviewed_count = len(result.interviews)
            
        except ValueError as e:
            # Simulationsumgebung nicht aktiv
            logger.warning(t("console.interviewApiCallFailed", error=e))
            result.summary = f"Interview fehlgeschlagen: {str(e)}. Die Simulationsumgebung ist möglicherweise geschlossen, bitte stellen Sie sicher, dass die OASIS-Umgebung läuft."
            return result
        except Exception as e:
            logger.error(t("console.interviewApiCallException", error=e))
            import traceback
            logger.error(traceback.format_exc())
            result.summary = f"Fehler während des Interviews: {str(e)}"
            return result
        
        # Step 6: Interviewzusammenfassung generieren
        if result.interviews:
            result.summary = self._generate_interview_summary(
                interviews=result.interviews,
                interview_requirement=interview_requirement
            )
        
        logger.info(t("console.interviewAgentsComplete", count=result.interviewed_count))
        return result
    
    @staticmethod
    def _clean_tool_call_response(response: str) -> str:
        """JSON-Werkzeugaufruf-Verpackung in Agent-Antworten bereinigen, tatsächlichen Inhalt extrahieren"""
        if not response or not response.strip().startswith('{'):
            return response
        text = response.strip()
        if 'tool_name' not in text[:80]:
            return response
        import re as _re
        try:
            data = json.loads(text)
            if isinstance(data, dict) and 'arguments' in data:
                for key in ('content', 'text', 'body', 'message', 'reply'):
                    if key in data['arguments']:
                        return str(data['arguments'][key])
        except (json.JSONDecodeError, KeyError, TypeError):
            match = _re.search(r'"content"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
            if match:
                return match.group(1).replace('\\n', '\n').replace('\\"', '"')
        return response

    def _load_agent_profiles(self, simulation_id: str) -> List[Dict[str, Any]]:
        """Agent-Persona-Dateien der Simulation laden"""
        import os
        import csv
        
        # Pfad zur Persona-Datei erstellen
        sim_dir = os.path.join(
            os.path.dirname(__file__), 
            f'../../uploads/simulations/{simulation_id}'
        )
        
        profiles = []
        
        # Bevorzugt Reddit JSON-Format lesen
        reddit_profile_path = os.path.join(sim_dir, "reddit_profiles.json")
        if os.path.exists(reddit_profile_path):
            try:
                with open(reddit_profile_path, 'r', encoding='utf-8') as f:
                    profiles = json.load(f)
                logger.info(t("console.loadedRedditProfiles", count=len(profiles)))
                return profiles
            except Exception as e:
                logger.warning(t("console.readRedditProfilesFailed", error=e))
        
        # Twitter CSV-Format lesen versuchen
        twitter_profile_path = os.path.join(sim_dir, "twitter_profiles.csv")
        if os.path.exists(twitter_profile_path):
            try:
                with open(twitter_profile_path, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        # CSV-Format in einheitliches Format umwandeln
                        profiles.append({
                            "realname": row.get("name", ""),
                            "username": row.get("username", ""),
                            "bio": row.get("description", ""),
                            "persona": row.get("user_char", ""),
                            "profession": "Unbekannt"
                        })
                logger.info(t("console.loadedTwitterProfiles", count=len(profiles)))
                return profiles
            except Exception as e:
                logger.warning(t("console.readTwitterProfilesFailed", error=e))
        
        return profiles
    
    def _select_agents_for_interview(
        self,
        profiles: List[Dict[str, Any]],
        interview_requirement: str,
        simulation_requirement: str,
        max_agents: int
    ) -> tuple:
        """
        LLM verwenden um zu interviewende Agents auszuwählen
        
        Returns:
            tuple: (selected_agents, selected_indices, reasoning)
                - selected_agents: Vollständige Informationsliste der ausgewählten Agents
                - selected_indices: Indexliste der ausgewählten Agents (für API-Aufrufe)
                - reasoning: Auswahlbegründung
        """
        
        # Agent-Zusammenfassungsliste erstellen
        agent_summaries = []
        for i, profile in enumerate(profiles):
            summary = {
                "index": i,
                "name": profile.get("realname", profile.get("username", f"Agent_{i}")),
                "profession": profile.get("profession", "Unbekannt"),
                "bio": profile.get("bio", "")[:200],
                "interested_topics": profile.get("interested_topics", [])
            }
            agent_summaries.append(summary)
        
        system_prompt = """Du bist ein professioneller Interview-Planungsexperte. Deine Aufgabe ist es, basierend auf den Interviewanforderungen die am besten geeigneten Interviewpartner aus der Liste der Simulations-Agents auszuwählen.

Auswahlkriterien:
1. Die Identität/der Beruf des Agents ist relevant für das Interviewthema
2. Der Agent könnte einzigartige oder wertvolle Standpunkte haben
3. Vielfältige Perspektiven auswählen (z.B.: Befürworter, Gegner, Neutrale, Fachleute usw.)
4. Rollen bevorzugen, die direkt mit dem Ereignis zusammenhängen

Rückgabe im JSON-Format:
{
    "selected_indices": [Indexliste der ausgewählten Agents],
    "reasoning": "Begründung der Auswahl"
}"""

        user_prompt = f"""Interviewanforderung:
{interview_requirement}

Simulationshintergrund:
{simulation_requirement if simulation_requirement else "Nicht angegeben"}

Verfügbare Agent-Liste (insgesamt {len(agent_summaries)} Stück):
{json.dumps(agent_summaries, ensure_ascii=False, indent=2)}

Bitte wählen Sie maximal {max_agents} am besten geeignete Agents für das Interview aus und begründen Sie die Auswahl."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3
            )
            
            selected_indices = response.get("selected_indices", [])[:max_agents]
            reasoning = response.get("reasoning", "Basierend auf Relevanz automatisch ausgewählt")
            
            # Vollständige Informationen der ausgewählten Agents abrufen
            selected_agents = []
            valid_indices = []
            for idx in selected_indices:
                if 0 <= idx < len(profiles):
                    selected_agents.append(profiles[idx])
                    valid_indices.append(idx)
            
            return selected_agents, valid_indices, reasoning
            
        except Exception as e:
            logger.warning(t("console.llmSelectAgentFailed", error=e))
            # Fallback: Die ersten N auswählen
            selected = profiles[:max_agents]
            indices = list(range(min(max_agents, len(profiles))))
            return selected, indices, "Standard-Auswahlstrategie verwendet"
    
    def _generate_interview_questions(
        self,
        interview_requirement: str,
        simulation_requirement: str,
        selected_agents: List[Dict[str, Any]]
    ) -> List[str]:
        """Interviewfragen mit LLM generieren"""
        
        agent_roles = [a.get("profession", "Unbekannt") for a in selected_agents]
        
        system_prompt = """Du bist ein professioneller Journalist/Interviewer. Generiere basierend auf den Interviewanforderungen 3-5 Tiefeninterviewfragen.

Fraganforderungen:
1. Offene Fragen, die detaillierte Antworten fördern
2. Auf verschiedene Rollen könnten unterschiedliche Antworten zutreffen
3. Fakten, Meinungen, Gefühle und weitere Dimensionen abdecken
4. Natürliche Sprache, wie in einem echten Interview
5. Jede Frage auf maximal 50 Wörter begrenzen, kurz und prägnant
6. Direkt fragen, keine Hintergrundinformationen oder Präfixe einbauen

Rückgabe im JSON-Format: {"questions": ["Frage1", "Frage2", ...]}"""

        user_prompt = f"""Interviewanforderung: {interview_requirement}

Simulationshintergrund: {simulation_requirement if simulation_requirement else "Nicht angegeben"}

Rollen der Interviewpartner: {', '.join(agent_roles)}

Bitte generieren Sie 3-5 Interviewfragen."""

        try:
            response = self.llm.chat_json(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.5
            )
            
            return response.get("questions", [f"Was denken Sie über {interview_requirement}?"])
            
        except Exception as e:
            logger.warning(t("console.generateInterviewQuestionsFailed", error=e))
            return [
                f"Was ist Ihre Meinung zu {interview_requirement}?",
                "Welche Auswirkungen hat diese Angelegenheit auf Sie oder die Gruppe, die Sie vertreten?",
                "Wie sollte dieses Problem Ihrer Meinung nach gelöst oder verbessert werden?"
            ]
    
    def _generate_interview_summary(
        self,
        interviews: List[AgentInterview],
        interview_requirement: str
    ) -> str:
        """Interviewzusammenfassung generieren"""
        
        if not interviews:
            return "Keine Interviews abgeschlossen"
        
        # Alle Interviewinhalte sammeln
        interview_texts = []
        for interview in interviews:
            interview_texts.append(f"【{interview.agent_name}（{interview.agent_role}）】\n{interview.response[:500]}")
        
        quote_instruction = "引用受访者原话时使用中文引号「」" if get_locale() == 'zh' else 'Use quotation marks "" when quoting interviewees'
        system_prompt = f"""Du bist ein professioneller Nachrichtenredakteur. Erstelle basierend auf den Antworten mehrerer Befragter eine Interviewzusammenfassung.

Anforderungen an die Zusammenfassung:
1. Hauptstandpunkte aller Parteien herausarbeiten
2. Konsens und Differenzen in den Standpunkten aufzeigen
3. Wertvolle Zitate hervorheben
4. Objektiv und neutral, ohne Parteinahme
5. Auf maximal 1000 Wörter begrenzen

Formatvorgaben (müssen eingehalten werden):
- Reine Textabsätze verwenden, verschiedene Teile durch Leerzeilen trennen
- Keine Markdown-Überschriften verwenden (wie #, ##, ###)
- Keine Trennlinien verwenden (wie ---, ***)
- {quote_instruction}
- **Fettdruck** für Schlüsselwörter ist erlaubt, aber keine andere Markdown-Syntax"""

        user_prompt = f"""Interviewthema: {interview_requirement}

Interviewinhalt:
{"".join(interview_texts)}

Bitte erstellen Sie eine Interviewzusammenfassung."""

        try:
            summary = self.llm.chat(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )
            return summary
            
        except Exception as e:
            logger.warning(t("console.generateInterviewSummaryFailed", error=e))
            # Fallback: Einfache Zusammenfügung
            return f"Insgesamt {len(interviews)} Befragte interviewt, darunter: " + ", ".join([i.agent_name for i in interviews])
