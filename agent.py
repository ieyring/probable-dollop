NEW_CONCEPT = "new_concept"
PROPERTY = "property"
SUBCONCEPT = "subconcept"
METACONCEPT = "metaconcept"
EXISTING_CONCEPT = "existing_concept"


import json, re
from typing import Dict, List

import os
import time
import uuid
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq, APIStatusError

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY ist nicht gesetzt.")

client = Groq(api_key=GROQ_API_KEY)

MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "10"))

# Sandbox & Limits
ALLOWED_BASE_PATH = os.getenv("AGENT_SANDBOX_PATH", "sandbox")
MAX_FILE_SIZE = int(os.getenv("AGENT_MAX_FILE_SIZE", "2_000_000"))       # Bytes
MAX_TEXT_LENGTH = int(os.getenv("AGENT_MAX_TEXT_LENGTH", "200_000"))     # Zeichen

# Groq-Settings
GROQ_TIMEOUT = float(os.getenv("AGENT_GROQ_TIMEOUT", "30.0"))
GROQ_MAX_RETRIES = int(os.getenv("AGENT_GROQ_MAX_RETRIES", "3"))
GROQ_RETRY_BACKOFF = float(os.getenv("AGENT_GROQ_RETRY_BACKOFF", "1.5"))

class AuditLogger:
    def __init__(self, path="agent_audit.log"):
        self.path = path

    def log(self, event_type: str, data: dict):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event": event_type,
            "data": data,
        }
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            # Im Produktionsfall evtl. an stderr oder externes Logging
            pass


audit = AuditLogger()


def new_request_id() -> str:
    return str(uuid.uuid4())

# ---------------------------------------------------------
# Groq Helper mit Retry & Timeout
# ---------------------------------------------------------
def estimate_tokens(messages: list, chars_per_token: float = 1.5) -> int:
    """
    Grobe Token-Schätzung basierend auf der Annahme:
    1 Token ≈ 3–4 Zeichen (Standard für Llama/Groq Modelle).

    messages: Liste von Chat-Nachrichten im Format:
        [{"role": "...", "content": "..."}]

    chars_per_token: Faktor zur Feinjustierung (Default 3.5)

    Rückgabe: geschätzte Tokenanzahl (int)
    """
    total_chars = 0

    for msg in messages:
        if isinstance(msg, dict): 
            role = msg.get("role", "") or ""
            content = msg.get("content", "") or ""
        else: # ChatCompletionMessage oder ähnliches 
            role = getattr(msg, "role", "") or ""
            content = getattr(msg, "content", "") or ""
        # Rollen + Content zählen, weil LLMs beides tokenisieren
        total_chars += len(role) + len(content)

    estimated = int(total_chars / chars_per_token)
    return estimated

def call_groq_chat(model: str, messages: list, tools=None, tool_choice=None):
    # 0. Token-Schätzung vor dem Request 
    estimated_tokens = estimate_tokens(messages) 
    audit.log("token_estimate", {"estimated_tokens": estimated_tokens})

    last_error = None

    for attempt in range(1, GROQ_MAX_RETRIES + 1):
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "timeout": GROQ_TIMEOUT,
            }

            # Nur wenn Tools gesetzt sind, auch tools/tool_choice mitschicken
            if tools is not None:
                kwargs["tools"] = tools
                if tool_choice is not None:
                    kwargs["tool_choice"] = tool_choice

            resp = client.chat.completions.create(**kwargs)
            usage = getattr(resp, "usage", None) 
            if usage: 
                factor = usage.prompt_tokens / estimated_tokens 
                audit.log("token_factor", {"factor": factor})
                audit.log("token_usage", { 
                    "prompt_tokens": usage.prompt_tokens, 
                    "completion_tokens": usage.completion_tokens, 
                    "total_tokens": usage.total_tokens 
                    })
            return resp

        except APIStatusError as e:
            last_error = e
            if e.status_code in (429, 500, 502, 503, 504) and attempt < GROQ_MAX_RETRIES:
                time.sleep(GROQ_RETRY_BACKOFF * attempt)
                continue
            raise
        except Exception as e:
            last_error = e
            if attempt < GROQ_MAX_RETRIES:
                time.sleep(GROQ_RETRY_BACKOFF * attempt)
                continue
            raise

    raise last_error or RuntimeError("Unbekannter Fehler im LLM-Aufruf")

# ---------------------------------------------------------
# Speichern
# ---------------------------------------------------------

BASE = "data"

def save_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

import os, json
from datetime import datetime

def save_versioned(path_base: str, obj_id: str, data: dict):
    """
    Speichert ein JSON-Objekt versioniert.
    path_base: z.B. "data/world_model/concepts"
    obj_id: z.B. "concept_Alois_Franz_von_Waibling"
    data: das zu speichernde Objekt (ohne version/previous_versions)
    """

    os.makedirs(path_base, exist_ok=True)

    # existierende Versionen finden
    existing = [
        f for f in os.listdir(path_base)
        if f.startswith(obj_id + "_v") and f.endswith(".json")
    ]

    if existing:
        versions = [
            int(f.split("_v")[1].split(".")[0])
            for f in existing
        ]
        new_version = max(versions) + 1
        prev = versions
    else:
        new_version = 1
        prev = []

    # Metadaten hinzufügen
    data_to_save = {
        **data,
        "id": obj_id,
        "version": new_version,
        "previous_versions": prev,
        "created_at": datetime.utcnow().isoformat()
    }

    # Datei speichern
    path = f"{path_base}/{obj_id}_v{new_version}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, ensure_ascii=False, indent=2)

    return data_to_save

from sentence_transformers import SentenceTransformer

class EmbeddingClient:
    def __init__(self, model="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model)

    def embed(self, text: str):
        return self.model.encode(text, normalize_embeddings=True)

import numpy as np

class EmbeddingIndex:
    def __init__(self):
        self.vectors = []
        self.labels = []

    def add(self, label: str, vector):
        self.labels.append(label)
        self.vectors.append(vector)

    def search(self, query_vector, top_k=3):
        if not self.vectors:
            return []

        matrix = np.vstack(self.vectors)
        sims = matrix @ query_vector  

        idx = np.argsort(-sims)[:top_k]

        return [(self.labels[i], float(sims[i])) for i in idx]

def build_index(world_model, embedder):
    index = EmbeddingIndex()

    for label, data in world_model.items():
        term = data.get("term", label)
        vec = embedder.embed(term)
        index.add(label, vec)

    return index

class LLMClient:
    def __call__(self, prompt: str) -> str:
        messages = [
            {"role": "system", "content": "Du bist ein präziser Extraktions-Agent. Gib ausschließlich JSON zurück."},
            {"role": "user", "content": prompt}
        ]

        resp = call_groq_chat(
            model="llama-3.3-70b-versatile",  
            messages=messages
        )

        return resp.choices[0].message.content

class IEModule:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def extract_terms(self, text: str) -> Dict:
        prompt = self._build_prompt(text)
        response = self.llm(prompt)
        return self._parse_response(response)

    def _build_prompt(self, text: str) -> str:
        return f"""
Extrahiere aus folgendem Text alle relevanten Begriffe ("terms").
Für jeden Begriff:

- term
- context (Person, Ort, Eigenschaft, Prozess, Norm, Kategorie, abstrakt)
- attributes (Liste)
- relations (Liste)
- negations (Liste von Begriffen, die explizit verneint werden, z.B. "keine Mango")
- is_proper_name (true/false)

Gib ausschließlich JSON zurück:

{{
  "terms": [
    {{
      "term": "...",
      "context": "...",
      "attributes": [...],
      "relations": [...],
      "negations": [...],
      "is_proper_name": true
    }}
  ]
}}

Text:
{text}
"""

    def _parse_response(self, response: str) -> Dict:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            raise ValueError("LLM lieferte kein JSON")
        data = json.loads(match.group(0))
        if "terms" not in data:
            raise ValueError("JSON enthält kein 'terms'-Feld")
        for t in data["terms"]:
            t.setdefault("context", "unknown")
            t.setdefault("attributes", [])
            t.setdefault("relations", [])
            t.setdefault("negations", [])
            t.setdefault("is_proper_name", False)
        return data


class ConceptClassifier:
    ADJ_SUFFIXES = ("ig", "lich", "bar", "los", "isch", "haft", "voll", "sam", "frei")

    def __init__(self, world_model, llm=None, embedder=EmbeddingClient):
        # world_model ist ein Dict von Label → Datenobjekt
        self.world_model = world_model
        self.llm = llm  # optional, zur späteren LLM‑gestützten Klassifikation
        self.embedder = embedder

        if embedder:
            self.embedding_index = build_index(world_model, embedder)

    def classify(self, term_obj: Dict) -> Dict:
        term = term_obj["term"]
        context = term_obj.get("context", "").lower()
        is_proper = term_obj.get("is_proper_name", False)
        relations = term_obj.get("relations", [])
    
        # 1. PRÜFUNG: Steht in den Relationen etwas, das wir schon als Konzept kennen?
        for rel in relations:
            if rel in self.world_model:
                # Volltreffer! Wir haben einen Parent über die Relationen gefunden
                return {"classification": SUBCONCEPT, "parent": rel}

        # Name proper nouns sollen in der Regel eigene Konzepte werden
        if is_proper:
            return {"classification": NEW_CONCEPT}

        # Kontextbasierte Heuristiken
        if context in ["eigenschaft", "charaktereigenschaft"]:
            if self._is_adjective(term):
                return {"classification": PROPERTY}
            # einfache Heuristik: alles kleingeschriebene Wort ohne Großbuchstaben
            if term and term[0].islower():
                return {"classification": PROPERTY}
            return {"classification": NEW_CONCEPT}

        if context in ["emotion", "prozess", "norm", "kategorie", "abstrakt"]:
            # diese Typen behandeln wir aktuell als neue Konzepte
            return {"classification": NEW_CONCEPT}

        # Suche nach möglichem Elternbegriff im existierenden Weltmodell
        parent = self._find_parent_concept(term)
        if parent:
            return {"classification": SUBCONCEPT, "parent": parent}

        # Prüfe, ob der Term ein Metakonzept sein sollte
        if self._should_be_metaconcept(term):
            return {"classification": METACONCEPT}

        
        if self.llm is not None:
            try:
                return self._llm_classify(term_obj)
            except Exception:
                pass

        # default
        return {"classification": NEW_CONCEPT}

    def _is_adjective(self, term: str) -> bool:
        t = term.lower()
        if t and t[0].islower():
            return any(t.endswith(suf) for suf in self.ADJ_SUFFIXES)
        return False

    def _find_parent_concept(self, term: str):
        if not hasattr(self, "embedding_index"):
            return None

        query_vec = self.embedder.embed(term)
        results = self.embedding_index.search(query_vec, top_k=3)

        for label, score in results:
            if label.lower() == term.lower():
                continue

            if score > 0.75:  # 🔥 critical threshold
                return label

        return None

    def _simple_find_parent_concept(self, term: str):
        # einfache stringbasierte Ähnlichkeit & Teilstrings
        lower = term.lower()
        candidates = []
        for data in self.world_model.values():
            existing = data.get("term", "").lower()
            if existing == lower:
                continue
            # nur echtes Containment, kein Gleichheit
            if lower in existing or existing in lower:
                candidates.append(data.get("term"))
        if candidates:
            # die kürzeste Übereinstimmung wählen (z.B. "Zahl" vor "Zahlenraum")
            return sorted(candidates, key=len)[0]

        # fuzzy matching mit difflib als Fallback
        from difflib import get_close_matches
        terms = [d.get("term", "") for d in self.world_model.values()]
        matches = get_close_matches(term, terms, n=1, cutoff=0.8)
        if matches and matches[0].lower() != lower:
            return matches[0]
        return None

    def _should_be_metaconcept(self, term: str) -> bool:
        lower = term.lower()
        words = lower.split()
        for w in words:
            count = 0
            for data in self.world_model.values():
                existing = data.get("term", "").lower()
                if existing != lower and w in existing:
                    count += 1
                if count >= 3:
                    return True
        return False

    def _llm_classify(self, term_obj: Dict) -> Dict:
        # kleiner Fallback auf LLM, falls heuristische Regeln nicht greifen
        prompt = f"""
Du bist ein Klassifikator für Begriffe im Weltmodell. Entscheide, ob der Begriff
als NEW_CONCEPT, PROPERTY, SUBCONCEPT (mit parent) oder METACONCEPT
abgelegt werden sollte. Antworte als JSON.

term: {term_obj.get('term')}
context: {term_obj.get('context')}
attributes: {term_obj.get('attributes')}
relations: {term_obj.get('relations')}
is_proper_name: {term_obj.get('is_proper_name')}

existierende Konzepte: {', '.join([d.get('term','') for d in self.world_model.values()][:50])}
"""
        resp = self.llm(prompt)
        try:
            data = json.loads(resp)
            # Mindestens Feld classification benötigt
            if "classification" in data:
                return data
        except Exception:
            pass
        return {"classification": NEW_CONCEPT}

class WorldModel(dict):
    def add_concept(self, label, data):
        self[label] = data

    def add_property(self, label, data):
        self[label] = data

    def add_subconcept(self, label, parent, data):
        self[label] = data
        self[label]["parent"] = parent

    def add_metaconcept(self, label, data):
        self[label] = data

class WorldModelUpdater:
    def __init__(self, world_model: WorldModel):
        self.world_model = world_model

    def _validate_and_fix(self, term_obj):
        """Bereinigt Datenfehler, bevor sie ins Dateisystem geschrieben werden."""
        term = term_obj.get("term", "")
        context = term_obj.get("context", "")
        relations = term_obj.get("relations", [])
        
        # 1. Eigenname-Check: Mango/Apfel sind keine Eigennamen
        # Heuristik: Wenn ein Kontext wie 'Frucht' oder 'Kategorie' existiert, 
        # ist es meist kein Proper Name (außer es ist eine Marke).
        if term_obj.get("is_proper_name") and context.lower() in ["frucht", "obst", "kategorie"]:
            term_obj["is_proper_name"] = False

        # 2. Parent-Auto-Discovery: Suche in context und relations
        if not term_obj.get("parent"):
            # Prüfe Kandidaten (Normalisiert auf Singular)
            candidates = [context] + relations
            for cand in candidates:
                if not cand: continue
                norm_cand = cand.rstrip('e').rstrip('n')
                
                # Wenn der Kandidat (oder die Singularform) im Weltmodell existiert
                if norm_cand in self.world_model and norm_cand != term:
                    term_obj["parent"] = norm_cand
                    term_obj["context"] = "Subkonzept"
                    break
        
        return term_obj
    
    def heal_entire_model(self):
        """Scannt alle vorhandenen JSONs und wendet die Fixes an."""
        actions = []
        for term, data in self.world_model.items():
            repaired_data = self._validate_and_fix(data.copy())
            if repaired_data != data:
                # Speichern der korrigierten Version
                self.world_model[term] = repaired_data
                # Hier save_versioned aufrufen je nach Typ...
                actions.append(f"Healed: {term}")
        return actions

    def apply_classification(self, term_obj, classification):
        term_obj = self._validate_and_fix(term_obj)
        term = term_obj["term"]
        kind = classification["classification"]

        # NEU: Fall für die Aktualisierung bestehender Konzepte
        if kind == EXISTING_CONCEPT:
            if term not in self.world_model:
                return {"action": "error", "reason": "term_not_found", "term": term}
            
            # Bestehende Daten holen
            current_data = self.world_model[term]
            # 1. Konflikt erkennen
            if self.detect_parent_conflict(term, classification):
                return self.resolve_parent_conflict(term, classification)
            
            # 2. Normaler Parent-Update (falls kein Konflikt)
            if "new_parent" in classification:
                current_data["parent"] = classification["new_parent"]
                current_data["context"] = "Subkonzept"
                save_versioned(
                    path_base="data/world_model/subconcepts",
                    obj_id=f"subconcept_{term}",
                    data=current_data
                )
                return {"action": "relinked_to_parent", "term": term, "parent": classification["new_parent"]}
            
            new_attr = classification.get("new_attribute")
                        
            if new_attr:
                # Attribut hinzufügen, falls noch nicht vorhanden
                if "attributes" not in current_data:
                    current_data["attributes"] = []
                
                if new_attr not in current_data["attributes"]:
                    current_data["attributes"].append(new_attr)
                    self.world_model[term] = current_data  # WICHTIG: Speicher-Objekt aktualisieren!
                    # Pfad bestimmen (wir müssen wissen, wo es liegt)
                    # Hier eine kleine Hilfslogik oder den Standard-Pfad nutzen:
                    path_map = {
                        "parent": "data/world_model/subconcepts", # Falls es ein Subconcept ist
                        "metaconcept": "data/world_model/metaconcepts"
                    }
                    # Standardmäßig in concepts speichern, wenn nichts anderes passt
                    base_path = "data/world_model/concepts"
                    if "parent" in current_data: base_path = "data/world_model/subconcepts"
                    
                    # Versioniert speichern
                    save_versioned(
                        path_base=base_path,
                        obj_id=f"concept_{term}",
                        data=current_data
                    )
                    return {"action": "update_concept", "term": term, "added_attribute": new_attr}
            
            return {"action": "no_change", "term": term}
        
        # vermeide Überschreiben vorhandener Begriffe – wenn schon da, nichts tun
        if term in self.world_model:
            return {"action": "exists", "term": term}

        if kind == NEW_CONCEPT:
            self.world_model.add_concept(term, term_obj)

            save_versioned(
                path_base="data/world_model/concepts",
                obj_id=f"concept_{term}",
                data=term_obj
            )

            return {"action": "create_concept", "term": term}


        if kind == PROPERTY:
            self.world_model.add_property(term, term_obj)

            save_versioned(
                path_base="data/world_model/properties",
                obj_id=f"property_{term}",
                data=term_obj
            )

            return {"action": "create_property", "term": term}


        if kind == SUBCONCEPT:
            parent = classification["parent"]
            self.world_model.add_subconcept(term, parent, term_obj)

            save_versioned(
                path_base="data/world_model/subconcepts",
                obj_id=f"subconcept_{term}",
                data={**term_obj, "parent": parent}
            )

            return {"action": "create_subconcept", "term": term, "parent": parent}


        if kind == METACONCEPT:
            self.world_model.add_metaconcept(term, term_obj)

            save_versioned(
                path_base="data/world_model/metaconcepts",
                obj_id=f"metaconcept_{term}",
                data=term_obj
            )

            return {"action": "create_metaconcept", "term": term}

    def apply_synthesis(self, suggestions: List[Dict]) -> List[Dict]:
        """Ähnlich wie `apply_classification`, aber nimmt eine Liste von
        Vorschlägen (jeweils mit keys `term`, `classification` und optional
        `parent`). Gibt Aktionen zurück. Überspringt bereits vorhandene Begriffe.
        """
        results = []
        for s in suggestions:
            term = s.get("term")
            kind = s.get("classification")
            
            # Nur überspringen, wenn es kein Update eines bestehenden Konzepts ist
            if term in self.world_model and kind != EXISTING_CONCEPT:
                results.append({"action": "exists", "term": term})
                continue
            
            # Bei EXISTING_CONCEPT erlauben wir den Aufruf, auch wenn der Term existiert
            proc = self.apply_classification({"term": term}, s)
            results.append(proc)
        return results

    def detect_parent_conflict(self, term, extracted_info):
        """Prüft, ob der Text eine bestehende Parent-Relation negiert."""
        current_parent = self.world_model[term].get("parent")
        negations = extracted_info.get("negations", [])
        
        if current_parent and current_parent in negations:
            return True
        return False
    
    def resolve_parent_conflict(self, term, extracted_info):
        """Ersetzt oder entfernt einen Parent, wenn der Text das explizit verlangt."""
        current_parent = self.world_model[term].get("parent")
        new_parent = extracted_info.get("new_parent")

        # Fall A: Neuer Parent vorhanden → ersetzen
        if new_parent:
            self.world_model[term]["parent"] = new_parent
            self.world_model[term]["context"] = "Subkonzept"
            return {"action": "parent_replaced", "term": term, "old": current_parent, "new": new_parent}

        # Fall B: Kein neuer Parent → Parent löschen
        self.world_model[term]["parent"] = None
        return {"action": "parent_removed", "term": term, "old": current_parent}

def load_world_model(path="data/world_model"):
    wm = WorldModel()

    def load_latest(subpath, add_fn):
        full = os.path.join(path, subpath)
        if not os.path.exists(full):
            return

        # alle Dateien gruppieren nach obj_id
        grouped = {}
        for fname in os.listdir(full):
            if not fname.endswith(".json"):
                continue
            obj_id = fname.split("_v")[0]
            grouped.setdefault(obj_id, []).append(fname)

        # pro obj_id die höchste Version laden
        for obj_id, files in grouped.items():
            latest = sorted(files)[-1]
            with open(os.path.join(full, latest), "r", encoding="utf-8") as f:
                data = json.load(f)
                label = data["id"].split("_", 1)[1]  # z.B. concept_Alois → Alois
                add_fn(label, data)

    load_latest("concepts", wm.add_concept)
    load_latest("properties", wm.add_property)
    load_latest("subconcepts", lambda label, data: wm.add_subconcept(label, data.get("parent"), data))
    load_latest("metaconcepts", wm.add_metaconcept)

    return wm

class WorldModelAgent:
    def __init__(self, llm: LLMClient, world_model, embedder):
        self.ie = IEModule(llm)
        # Übergebe LLM auch an den Klassifikator für Fallbacks
        self.classifier = ConceptClassifier(world_model, llm=llm,embedder=embedder)
        self.updater = WorldModelUpdater(world_model)

    def process_text(self, text: str) -> List[Dict]:
        ie_result = self.ie.extract_terms(text)
        obs_id = new_request_id()

        save_versioned(
            path_base="data/observations",
            obj_id=f"obs_{obs_id}",
            data={
                "source_text": text,
                "terms": ie_result["terms"]
            }
        )


        actions = []
        for term_obj in ie_result["terms"]:
            classification = self.classifier.classify(term_obj)
            action = self.updater.apply_classification(term_obj, classification)
            actions.append(action)

        save_versioned(
            path_base="data/interpretations",
            obj_id=f"interp_{obs_id}",
            data={
                "observation_id": f"obs_{obs_id}",
                "classifications": actions
            }
        )
    
        return actions

class ConceptSynthesizer:
    """Erzeugt strukturierte Vorschläge basierend auf vorhandenen JSON‑Dateien.

    Läuft durch das geladene Weltmodell und identifiziert
    - wiederkehrende Attribute → mögliche PROPERTYs
    - wiederkehrende Relationsbegriffe → neue Konzepte
    - Subkonzepte anhand von Teilstrings/fuzzy Match
    - Metakonzepte (Wörter, die in vielen anderen Termen auftauchen)

    Die Methode `synthesize` liefert eine Liste von Vorschlägen, die dann
    vom Updater übernommen werden können.
    """
    def __init__(self, world_model: WorldModel,embedder:EmbeddingClient):
        self.world_model = world_model
        self.embedder = embedder
        if embedder:
            self.embedding_index = build_index(world_model, embedder)

    def synthesize(self) -> List[Dict]:
        suggestions: List[Dict] = []

        # Zähle Attribute und Relationsbegriffe
        attr_counts: Dict[str,int] = {}
        rel_counts: Dict[str,int] = {}
        for data in self.world_model.values():
            for a in data.get("attributes", []):
                attr_counts[a] = attr_counts.get(a, 0) + 1
            for r in data.get("relations", []):
                rel_counts[r] = rel_counts.get(r, 0) + 1

        # Vorschläge aus häufigen Attributen
        for term, cnt in attr_counts.items():
            if cnt >= 2 and term not in self.world_model:
                suggestions.append({"term": term, "classification": PROPERTY})

        # Relationsbegriffe als neue Konzepte vorschlagen
        for term, cnt in rel_counts.items():
            if cnt >= 2 and term not in self.world_model:
                suggestions.append({"term": term, "classification": NEW_CONCEPT})

        # Subkonzepte und Metakonzepte
        terms = [d.get("term", "") for d in self.world_model.values()]
        clf = ConceptClassifier(self.world_model,self.embedder)
        for term in terms:
            parent = clf._find_parent_concept(term)
            if parent and term not in self.world_model:
                suggestions.append({"term": term, "classification": SUBCONCEPT, "parent": parent})

        # Metakonzepte: häufige Substrings in mehreren Begriffen
        substr_counts: Dict[str,int] = {}
        lowered = [t.lower() for t in terms]
        for t in lowered:
            length = len(t)
            for i in range(length):
                for j in range(i+4, length+1):  # substrings mindestens 4 Zeichen
                    sub = t[i:j]
                    substr_counts[sub] = substr_counts.get(sub, 0) + 1
        for sub, cnt in substr_counts.items():
            if cnt >= 3 and sub not in self.world_model:
                suggestions.append({"term": sub, "classification": METACONCEPT})

        # eindeutige Begriffe zurückgeben
        unique = {}
        for s in suggestions:
            unique.setdefault(s["term"], s)
        return list(unique.values())

    def synthesize_shared_attributes(self):
        # Gruppiere alle Konzepte nach ihrem Parent
        groups = {}
        for term, data in self.world_model.items():
            parent = data.get("parent")
            if parent:
                groups.setdefault(parent, []).append(data)

        for parent, children in groups.items():
            if len(children) < 3: continue # Erst ab 3 Beispielen abstrahieren
            
            # Zähle, welche Attribute bei ALLEN (oder 80%) der Kinder vorkommen
            all_attrs = [set(c.get("attributes", [])) for c in children]
            common = set.intersection(*all_attrs) if all_attrs else set()
            
            for attr in common:
                # Vorschlag: Dieses Attribut ist ein definierendes Merkmal des Parents!
                print(f"Abstraktion: {attr} scheint ein Kernmerkmal von {parent} zu sein.")

    def synthesize_parent_properties(self) -> List[Dict]:
        """Findet gemeinsame Attribute bei Kindern und schlägt sie für den Parent vor."""
        suggestions = []
        # 1. Gruppiere Kinder nach Parent
        parent_to_children = {}
        for term, data in self.world_model.items():
            parent = data.get("parent")
            if parent:
                parent_to_children.setdefault(parent, []).append(data)

        # 2. Analysiere jede Gruppe
        for parent, children in parent_to_children.items():
            if len(children) < 2: continue  # Braucht mindestens 2 Beispiele
                
            # Schnittmenge der Attribute finden
            all_attr_sets = [set(c.get("attributes", [])) for c in children]
            common_attrs = set.intersection(*all_attr_sets)
                
            for attr in common_attrs:
                # Wenn der Parent das Attribut noch nicht hat: Vorschlag!
                parent_data = self.world_model.get(parent, {})
                if attr not in parent_data.get("attributes", []):
                    suggestions.append({
                        "term": parent, 
                        "classification": EXISTING_CONCEPT, # Update statt Neu
                        "new_attribute": attr,
                        "reason": f"Abstraktion von {len(children)} Unterkonzepten"
                    })
        return suggestions

    def relink_orphans(self) -> List[Dict]:
        suggestions = []

        for term, data in self.world_model.items():
            if data.get("parent"):
                continue

            # 🔥 Embedding-basierte Suche
            query_vec = self.embedder.embed(term)
            results = self.embedding_index.search(query_vec, top_k=3)

            for candidate, score in results:
                if candidate == term:
                    continue

                if score > 0.75:
                    suggestions.append({
                        "term": term,
                        "classification": EXISTING_CONCEPT,
                        "new_parent": candidate,
                        "confidence": score,
                        "reason": f"Embedding similarity ({score:.2f})"
                    })
                    break

        return suggestions

def get_effective_properties(world_model: dict, term: str) -> dict:
    """
    Sammelt alle Attribute und Relationen eines Begriffs rekursiv 
    über die gesamte Hierarchie ein.
    """
    # 1. Startpunkt finden
    data = world_model.get(term)
    if not data:
        return {"term": term, "attributes": [], "relations": [], "not_found": True}

    # 2. Lokale Daten extrahieren
    all_attributes = set(data.get("attributes", []))
    all_relations = set(data.get("relations", []))
    
    # 3. Rekursion: Wenn ein Parent existiert, steige nach oben
    parent_name = data.get("parent")
    if parent_name and parent_name in world_model:
        parent_properties = get_effective_properties(world_model, parent_name)
        
        # Merge die Sets (verhindert Dubletten)
        all_attributes.update(parent_properties["attributes"])
        all_relations.update(parent_properties["relations"])

    return {
        "term": term,
        "attributes": list(all_attributes),
        "relations": list(all_relations),
        "hierarchy_path": ([term] + parent_properties.get("hierarchy_path", []) 
                           if parent_name else [term])
    }
#print("ACTIONS:")
#for a in actions:
#    print(a)

#print("\nWORLD MODEL:")
#print(json.dumps(world_model, indent=2, ensure_ascii=False))
