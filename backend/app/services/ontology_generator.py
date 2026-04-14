"""
Ontologie-Generierungsdienst
Schnittstelle 1: Textinhalte analysieren und Entitäts- sowie Beziehungstypdefinitionen für soziale Simulationen generieren
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional
from ..utils.llm_client import LLMClient
from ..utils.locale import get_language_instruction

logger = logging.getLogger(__name__)


def _to_pascal_case(name: str) -> str:
    """Konvertiert einen Namen beliebigen Formats in PascalCase (z.B. 'works_for' -> 'WorksFor', 'person' -> 'Person')"""
    # Nach nicht-alphanumerischen Zeichen aufteilen
    parts = re.split(r'[^a-zA-Z0-9]+', name)
    # Zusätzlich an camelCase-Grenzen aufteilen (z.B. 'camelCase' -> ['camel', 'Case'])
    words = []
    for part in parts:
        words.extend(re.sub(r'([a-z])([A-Z])', r'\1_\2', part).split('_'))
    # Jeden Wortanfang großschreiben, leere Strings herausfiltern
    result = ''.join(word.capitalize() for word in words if word)
    return result if result else 'Unknown'


# System-Prompt für die Ontologie-Generierung
ONTOLOGY_SYSTEM_PROMPT = """Du bist ein professioneller Experte für Wissensgraphen-Ontologie-Design. Deine Aufgabe ist es, den gegebenen Textinhalt und die Simulationsanforderungen zu analysieren und Entitätstypen sowie Beziehungstypen zu entwerfen, die für die **Simulation von Social-Media-Meinungsbildung** geeignet sind.

**Wichtig: Du musst gültige JSON-formatierte Daten ausgeben und keinen anderen Inhalt.**

## Kernaufgabe – Hintergrund

Wir bauen ein **Simulationssystem für Social-Media-Meinungsbildung** auf. In diesem System:
- Jede Entität ist ein "Konto" oder "Akteur", der in sozialen Medien Meinungen äußern, interagieren und Informationen verbreiten kann
- Entitäten beeinflussen sich gegenseitig, teilen, kommentieren und reagieren aufeinander
- Wir müssen die Reaktionen und Informationsverbreitungswege der verschiedenen Parteien bei Meinungsereignissen simulieren

Daher **müssen Entitäten real existierende Akteure sein, die in sozialen Medien Meinungen äußern und interagieren können**:

**Möglich sind**:
- Konkrete Einzelpersonen (öffentliche Persönlichkeiten, Beteiligte, Meinungsführer, Fachexperten, gewöhnliche Menschen)
- Unternehmen und Firmen (einschließlich ihrer offiziellen Konten)
- Organisationen und Institutionen (Universitäten, Verbände, NGOs, Gewerkschaften usw.)
- Regierungsbehörden, Aufsichtsbehörden
- Medienorganisationen (Zeitungen, Fernsehsender, Selbstmedien, Websites)
- Social-Media-Plattformen selbst
- Vertreter bestimmter Gruppen (z.B. Alumni-Vereinigungen, Fangruppen, Interessenvertretungen usw.)

**Nicht möglich sind**:
- Abstrakte Konzepte (wie "Meinungsbildung", "Emotionen", "Trends")
- Themen/Gesprächsthemen (wie "akademische Integrität", "Bildungsreform")
- Meinungen/Haltungen (wie "Befürworter", "Gegner")

## Ausgabeformat

Bitte gib JSON-Format aus, das die folgende Struktur enthält:

```json
{
    "entity_types": [
        {
            "name": "Entitätstyp-Name (Englisch, PascalCase)",
            "description": "Kurze Beschreibung (Englisch, maximal 100 Zeichen)",
            "attributes": [
                {
                    "name": "Attributname (Englisch, snake_case)",
                    "type": "text",
                    "description": "Attributbeschreibung"
                }
            ],
            "examples": ["Beispielentität 1", "Beispielentität 2"]
        }
    ],
    "edge_types": [
        {
            "name": "Beziehungstyp-Name (Englisch, UPPER_SNAKE_CASE)",
            "description": "Kurze Beschreibung (Englisch, maximal 100 Zeichen)",
            "source_targets": [
                {"source": "Quell-Entitätstyp", "target": "Ziel-Entitätstyp"}
            ],
            "attributes": []
        }
    ],
    "analysis_summary": "Kurze Analysebeschreibung des Textinhalts"
}
```

## Designrichtlinien (Äußerst wichtig!)

### 1. Entitätstyp-Design – Muss strikt befolgt werden

**Anzahlvorgabe: Es müssen genau 10 Entitätstypen sein**

**Hierarchie-Anforderungen (muss sowohl spezifische als auch Auffangtypen enthalten)**:

Deine 10 Entitätstypen müssen die folgenden Ebenen umfassen:

A. **Auffangtypen (müssen enthalten sein, als letzte 2 in der Liste platziert)**:
   - `Person`: Auffangtyp für jede natürliche Person. Wenn eine Person keinem anderen spezifischeren Personentyp zugehört, wird sie hier eingeordnet.
   - `Organization`: Auffangtyp für jede Organisation. Wenn eine Organisation keinem anderen spezifischeren Organisationstyp zugehört, wird sie hier eingeordnet.

B. **Spezifische Typen (8, basierend auf dem Textinhalt entworfen)**:
   - Für die im Text vorkommenden Hauptakteure spezifischere Typen entwerfen
   - Beispiel: Wenn der Text ein akademisches Ereignis betrifft, können `Student`, `Professor`, `University` verwendet werden
   - Beispiel: Wenn der Text ein geschäftliches Ereignis betrifft, können `Company`, `CEO`, `Employee` verwendet werden

**Warum Auffangtypen benötigt werden**:
- Im Text erscheinen verschiedene Personen wie "Grundschullehrer", "Passant A", "ein bestimmter Internetnutzer"
- Wenn kein spezieller Typ passt, sollten sie `Person` zugeordnet werden
- Ebenso sollten kleine Organisationen, temporäre Gruppen usw. `Organization` zugeordnet werden

**Designprinzipien für spezifische Typen**:
- Aus dem Text häufig vorkommende oder wichtige Rollentypen identifizieren
- Jeder spezifische Typ sollte klare Grenzen haben, Überlappungen vermeiden
- Die Beschreibung muss den Unterschied zwischen diesem Typ und dem Auffangtyp klar erläutern

### 2. Beziehungstyp-Design

- Anzahl: 6-10
- Beziehungen sollten reale Verbindungen in Social-Media-Interaktionen widerspiegeln
- Stelle sicher, dass die source_targets der Beziehungen die von dir definierten Entitätstypen abdecken

### 3. Attribut-Design

- 1-3 Schlüsselattribute pro Entitätstyp
- **Hinweis**: Attributnamen dürfen nicht `name`, `uuid`, `group_id`, `created_at`, `summary` verwenden (diese sind Systemreservierungen)
- Empfohlen: `full_name`, `title`, `role`, `position`, `location`, `description` usw.

## Entitätstyp-Referenz

**Personenkategorie (spezifisch)**:
- Student: Student
- Professor: Professor/Gelehrter
- Journalist: Journalist
- Celebrity: Prominenter/Influencer
- Executive: Führungskraft
- Official: Regierungsbeamter
- Lawyer: Rechtsanwalt
- Doctor: Arzt

**Personenkategorie (Auffang)**:
- Person: Jede natürliche Person (wird verwendet, wenn keine der obigen spezifischen Typen zutrifft)

**Organisationskategorie (spezifisch)**:
- University: Hochschule
- Company: Unternehmen
- GovernmentAgency: Regierungsbehörde
- MediaOutlet: Medienorganisation
- Hospital: Krankenhaus
- School: Grund-/Mittelschule
- NGO: Nichtregierungsorganisation

**Organisationskategorie (Auffang)**:
- Organization: Jede Organisation (wird verwendet, wenn keine der obigen spezifischen Typen zutrifft)

## Beziehungstyp-Referenz

- WORKS_FOR: Arbeitet für
- STUDIES_AT: Studiert an
- AFFILIATED_WITH: Zugehörig zu
- REPRESENTS: Vertritt
- REGULATES: Reguliert
- REPORTS_ON: Berichtet über
- COMMENTS_ON: Kommentiert
- RESPONDS_TO: Reagiert auf
- SUPPORTS: Unterstützt
- OPPOSES: Lehnt ab
- COLLABORATES_WITH: Kooperiert mit
- COMPETES_WITH: Konkurriert mit
"""


class OntologyGenerator:
    """
    Ontologie-Generator
    Analysiert Textinhalte und generiert Entitäts- und Beziehungstypdefinitionen
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm_client = llm_client or LLMClient()
    
    def generate(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Ontologie-Definition generieren
        
        Args:
            document_texts: Liste der Dokumenttexte
            simulation_requirement: Beschreibung der Simulationsanforderungen
            additional_context: Zusätzlicher Kontext
            
        Returns:
            Ontologie-Definition (entity_types, edge_types usw.)
        """
        # Benutzernachricht erstellen
        user_message = self._build_user_message(
            document_texts, 
            simulation_requirement,
            additional_context
        )
        
        lang_instruction = get_language_instruction()
        system_prompt = f"{ONTOLOGY_SYSTEM_PROMPT}\n\n{lang_instruction}\nIMPORTANT: Entity type names MUST be in English PascalCase (e.g., 'PersonEntity', 'MediaOrganization'). Relationship type names MUST be in English UPPER_SNAKE_CASE (e.g., 'WORKS_FOR'). Attribute names MUST be in English snake_case. Only description fields and analysis_summary should use the specified language above."
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        # LLM aufrufen
        result = self.llm_client.chat_json(
            messages=messages,
            temperature=0.3,
            max_tokens=4096
        )
        
        # Validierung und Nachbearbeitung
        result = self._validate_and_process(result)
        
        return result
    
    # Maximale Textlänge für LLM (50.000 Zeichen)
    MAX_TEXT_LENGTH_FOR_LLM = 50000
    
    def _build_user_message(
        self,
        document_texts: List[str],
        simulation_requirement: str,
        additional_context: Optional[str]
    ) -> str:
        """Benutzernachricht erstellen"""
        
        # Texte zusammenführen
        combined_text = "\n\n---\n\n".join(document_texts)
        original_length = len(combined_text)
        
        # Bei über 50.000 Zeichen kürzen (betrifft nur LLM-Eingabe, nicht den Graphenaufbau)
        if len(combined_text) > self.MAX_TEXT_LENGTH_FOR_LLM:
            combined_text = combined_text[:self.MAX_TEXT_LENGTH_FOR_LLM]
            combined_text += f"\n\n...(Originaltext hat {original_length} Zeichen, die ersten {self.MAX_TEXT_LENGTH_FOR_LLM} Zeichen wurden für die Ontologie-Analyse verwendet)..."
        
        message = f"""## Simulationsanforderungen

{simulation_requirement}

## Dokumentinhalt

{combined_text}
"""
        
        if additional_context:
            message += f"""
## Zusätzliche Hinweise

{additional_context}
"""
        
        message += """
Bitte entwerfe anhand der obigen Inhalte Entitätstypen und Beziehungstypen, die für die Simulation sozialer Meinungsbildung geeignet sind.

**Zwingend einzuhaltende Regeln**:
1. Es müssen genau 10 Entitätstypen ausgegeben werden
2. Die letzten 2 müssen Auffangtypen sein: Person (Personen-Auffang) und Organization (Organisations-Auffang)
3. Die ersten 8 sind spezifische Typen, die auf dem Textinhalt basieren
4. Alle Entitätstypen müssen real existierende Akteure sein, die Meinungen äußern können, keine abstrakten Konzepte
5. Attributnamen dürfen keine Reservierungen wie name, uuid, group_id verwenden; stattdessen full_name, org_name usw. nutzen
"""
        
        return message
    
    def _validate_and_process(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Ergebnisse validieren und nachbearbeiten"""
        
        # Sicherstellen, dass erforderliche Felder vorhanden sind
        if "entity_types" not in result:
            result["entity_types"] = []
        if "edge_types" not in result:
            result["edge_types"] = []
        if "analysis_summary" not in result:
            result["analysis_summary"] = ""
        
        # Entitätstypen validieren
        # Mapping von Originalnamen zu PascalCase speichern, um spätere edge source_targets-Referenzen zu korrigieren
        entity_name_map = {}
        for entity in result["entity_types"]:
            # Entity-Name zwingend in PascalCase konvertieren (Zep API-Anforderung)
            if "name" in entity:
                original_name = entity["name"]
                entity["name"] = _to_pascal_case(original_name)
                if entity["name"] != original_name:
                    logger.warning(f"Entity type name '{original_name}' auto-converted to '{entity['name']}'")
                entity_name_map[original_name] = entity["name"]
            if "attributes" not in entity:
                entity["attributes"] = []
            if "examples" not in entity:
                entity["examples"] = []
            # Sicherstellen, dass die Beschreibung 100 Zeichen nicht überschreitet
            if len(entity.get("description", "")) > 100:
                entity["description"] = entity["description"][:97] + "..."
        
        # Beziehungstypen validieren
        for edge in result["edge_types"]:
            # Edge-Name zwingend in SCREAMING_SNAKE_CASE konvertieren (Zep API-Anforderung)
            if "name" in edge:
                original_name = edge["name"]
                edge["name"] = original_name.upper()
                if edge["name"] != original_name:
                    logger.warning(f"Edge type name '{original_name}' auto-converted to '{edge['name']}'")
            # Entitätsnamen-Referenzen in source_targets korrigieren, um Konsistenz mit dem konvertierten PascalCase zu gewährleisten
            for st in edge.get("source_targets", []):
                if st.get("source") in entity_name_map:
                    st["source"] = entity_name_map[st["source"]]
                if st.get("target") in entity_name_map:
                    st["target"] = entity_name_map[st["target"]]
            if "source_targets" not in edge:
                edge["source_targets"] = []
            if "attributes" not in edge:
                edge["attributes"] = []
            if len(edge.get("description", "")) > 100:
                edge["description"] = edge["description"][:97] + "..."
        
        # Zep API-Beschränkung: maximal 10 benutzerdefinierte Entitätstypen, maximal 10 benutzerdefinierte Kantentypen
        MAX_ENTITY_TYPES = 10
        MAX_EDGE_TYPES = 10

        # Deduplizierung: nach Name deduplizieren, erstes Vorkommen beibehalten
        seen_names = set()
        deduped = []
        for entity in result["entity_types"]:
            name = entity.get("name", "")
            if name and name not in seen_names:
                seen_names.add(name)
                deduped.append(entity)
            elif name in seen_names:
                logger.warning(f"Duplicate entity type '{name}' removed during validation")
        result["entity_types"] = deduped

        # Auffangtyp-Definitionen
        person_fallback = {
            "name": "Person",
            "description": "Any individual person not fitting other specific person types.",
            "attributes": [
                {"name": "full_name", "type": "text", "description": "Full name of the person"},
                {"name": "role", "type": "text", "description": "Role or occupation"}
            ],
            "examples": ["ordinary citizen", "anonymous netizen"]
        }
        
        organization_fallback = {
            "name": "Organization",
            "description": "Any organization not fitting other specific organization types.",
            "attributes": [
                {"name": "org_name", "type": "text", "description": "Name of the organization"},
                {"name": "org_type", "type": "text", "description": "Type of organization"}
            ],
            "examples": ["small business", "community group"]
        }
        
        # Prüfen, ob Auffangtypen bereits vorhanden sind
        entity_names = {e["name"] for e in result["entity_types"]}
        has_person = "Person" in entity_names
        has_organization = "Organization" in entity_names
        
        # Hinzuzufügende Auffangtypen
        fallbacks_to_add = []
        if not has_person:
            fallbacks_to_add.append(person_fallback)
        if not has_organization:
            fallbacks_to_add.append(organization_fallback)
        
        if fallbacks_to_add:
            current_count = len(result["entity_types"])
            needed_slots = len(fallbacks_to_add)
            
            # Wenn nach dem Hinzufügen mehr als 10 vorhanden wären, müssen einige bestehende Typen entfernt werden
            if current_count + needed_slots > MAX_ENTITY_TYPES:
                # Berechnen, wie viele entfernt werden müssen
                to_remove = current_count + needed_slots - MAX_ENTITY_TYPES
                # Vom Ende entfernen (wichtigere spezifische Typen am Anfang beibehalten)
                result["entity_types"] = result["entity_types"][:-to_remove]
            
            # Auffangtypen hinzufügen
            result["entity_types"].extend(fallbacks_to_add)
        
        # Abschließend sicherstellen, dass die Begrenzung nicht überschritten wird (defensive Programmierung)
        if len(result["entity_types"]) > MAX_ENTITY_TYPES:
            result["entity_types"] = result["entity_types"][:MAX_ENTITY_TYPES]
        
        if len(result["edge_types"]) > MAX_EDGE_TYPES:
            result["edge_types"] = result["edge_types"][:MAX_EDGE_TYPES]
        
        return result
    
    def generate_python_code(self, ontology: Dict[str, Any]) -> str:
        """
        Ontologie-Definition in Python-Code umwandeln (ähnlich wie ontology.py)
        
        Args:
            ontology: Ontologie-Definition
            
        Returns:
            Python-Code als String
        """
        code_lines = [
            '"""',
            'Benutzerdefinierte Entitätstyp-Definitionen',
            'Automatisch von MiroFish generiert, für soziale Meinungssimulation',
            '"""',
            '',
            'from pydantic import Field',
            'from zep_cloud.external_clients.ontology import EntityModel, EntityText, EdgeModel',
            '',
            '',
            '# ============== Entitätstyp-Definitionen ==============',
            '',
        ]
        
        # Entitätstypen generieren
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            desc = entity.get("description", f"A {name} entity.")
            
            code_lines.append(f'class {name}(EntityModel):')
            code_lines.append(f'    """{desc}"""')
            
            attrs = entity.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')
            
            code_lines.append('')
            code_lines.append('')
        
        code_lines.append('# ============== Beziehungstyp-Definitionen ==============')
        code_lines.append('')
        
        # Beziehungstypen generieren
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            # In PascalCase-Klassenname konvertieren
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            desc = edge.get("description", f"A {name} relationship.")
            
            code_lines.append(f'class {class_name}(EdgeModel):')
            code_lines.append(f'    """{desc}"""')
            
            attrs = edge.get("attributes", [])
            if attrs:
                for attr in attrs:
                    attr_name = attr["name"]
                    attr_desc = attr.get("description", attr_name)
                    code_lines.append(f'    {attr_name}: EntityText = Field(')
                    code_lines.append(f'        description="{attr_desc}",')
                    code_lines.append(f'        default=None')
                    code_lines.append(f'    )')
            else:
                code_lines.append('    pass')
            
            code_lines.append('')
            code_lines.append('')
        
        # Typwörterbuch generieren
        code_lines.append('# ============== Typkonfiguration ==============')
        code_lines.append('')
        code_lines.append('ENTITY_TYPES = {')
        for entity in ontology.get("entity_types", []):
            name = entity["name"]
            code_lines.append(f'    "{name}": {name},')
        code_lines.append('}')
        code_lines.append('')
        code_lines.append('EDGE_TYPES = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            class_name = ''.join(word.capitalize() for word in name.split('_'))
            code_lines.append(f'    "{name}": {class_name},')
        code_lines.append('}')
        code_lines.append('')
        
        # source_targets-Zuordnung für Kanten generieren
        code_lines.append('EDGE_SOURCE_TARGETS = {')
        for edge in ontology.get("edge_types", []):
            name = edge["name"]
            source_targets = edge.get("source_targets", [])
            if source_targets:
                st_list = ', '.join([
                    f'{{"source": "{st.get("source", "Entity")}", "target": "{st.get("target", "Entity")}"}}'
                    for st in source_targets
                ])
                code_lines.append(f'    "{name}": [{st_list}],')
        code_lines.append('}')
        
        return '\n'.join(code_lines)

