from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import sqlite3
import spacy
import os
import requests
import json
from dotenv import load_dotenv
import re

app = FastAPI()

load_dotenv("token.env")


# Together AI spezifische Variablen
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_MODEL = "mistralai/Mistral-Small-24B-Instruct-2501"  # Ein kostenloses Instruct-Modell (kannst du ändern)
TOGETHER_API_URL = "https://api.together.xyz/v1/completions"
TOGETHER_HEADERS = {
    "Authorization": f"Bearer {TOGETHER_API_KEY}",
    "Content-Type": "application/json"
}
print("TOGETHER_API_KEY:", TOGETHER_API_KEY)

FIELDS = ["city", "country", "price", "guests", "features", "type", "category", "amenities"]

# Prompt der and LLM gesendet wird.
def build_prompt(user_prompt: str) -> str:
    return f"""
    
ONLY OUTPUT RAW JSON. DO NOT USE MARKDOWN, CODE BLOCKS, OR EXTRA TEXT.

Translate non-English inputs into English first. Then extract structured travel data from the user’s request using the fields below.

**Target Fields:**

`city`, `country`, `price`, `guests`, `features`, `type`, `category`, `amenities`

**Instructions:**

1. **Understand Intent, Not Just Words**
    - Focus on what the user *wants*, not literal terms.
    - “I want to hear waves” → `features: "Beachfront"`
    - If a season/month is mentioned, reflect it in `type`.
2. **Assign Keywords to All Relevant Fields**
    - If a term fits multiple fields (e.g. “pool”), add to all: `features`, `amenities`, `type`.
    - `features` = physical traits
    - `type` = vibe, region, or travel style
3. **Smart Defaults**
    - Region only? Set country (e.g., “Tuscany” → `country: "Italy"` + `type: "Tuscany"`)
    - Group size hints: “couple” → 2, “family” → 4, “friends” → 5
    - Continent? Leave `country` empty, set continent in `type`
4. **Always Output All Fields**
    - `country` must be a valid country or blank
    - Include a natural `summary` at the end like:
        
        `"summary": "You’re looking for a cozy hideaway with mountain views, right?"`
        

**Final Format Example:**

(Do not change structure, just fill in with user’s intent)

```json
"price": 300,
"rating": 4.9,
"city": "Paro",
"amenities": "WiFi, Parking, Breakfast, Airport Shuttle",
"size": 75,
"category": "Hotel",
"features": "Mountain View, Balcony, Garden, Fitness Room, Spa",
"availability": "Available",
"country": "Bhutan",
"guests": 4,
"type": "Countryside, Adventure, Chilling, Luxury, Comfort, Asia",
"summary": "You are looking for a cozy place in Europe with sea view, right?"

```

Prompt: {user_prompt}
"""

# JSON wird erstellt und für mich zugreifbar
def extract_filters(user_prompt: str) -> dict:
    full_prompt = build_prompt(user_prompt)

    payload = {
        "model": TOGETHER_MODEL,
        "prompt": full_prompt,
        "max_tokens": 500,
        "temperature": 0.2,
    }

    print("\n🧠 Anfrage wird gesendet an Together AI...")
    try:
        response = requests.post(TOGETHER_API_URL, headers=TOGETHER_HEADERS, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors

        result = response.json()
        
        if 'choices' in result and len(result['choices']) > 0 and 'text' in result['choices'][0]:
            result_text = result['choices'][0]['text']
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            summary_match = re.search(r'"summary"\s*:\s*"([^"]+)"', result_text)

            filters = json.loads(json_match.group()) if json_match else {}
            summary = summary_match.group(1).strip() if summary_match else ""

            return {"filters": filters, "summary": summary}
        else:
            print(f"⚠️ LLM Antwort enthielt kein JSON: {result_text}")
            return {}

    except requests.exceptions.RequestException as e:
        print(f"❌ Fehler bei der Anfrage an Together AI: {e}")
        if response is not None:
            print(f"   Status Code: {response.status_code}")
            print(f"   Response Text: {response.text}")
        return {}
    except json.JSONDecodeError as e:
        print(f"⚠️ Fehler beim Decodieren der JSON-Antwort von Together AI: {e}")
        if response is not None:
            print(f"   Response Text: {response.text}")
        return {}


# JSON wird bereinigt und verwendbarer + erste Filter/Verschönerungen
def sanitize_filters(filters: dict) -> dict:
    """Stellt sicher, dass Filter datenbankbereit sind und wendet spezifische Bereinigungen an."""
    sanitized = {}
    if "guests" in filters and isinstance(filters["guests"], (int, str)) and str(filters["guests"]).isdigit():
        sanitized["guests"] = int(filters["guests"])
    else:
        sanitized["guests"] = 2  # Standardwert

    if "price" in filters:
        price_value = filters["price"]
        min_price = None
        max_price = None

        if isinstance(price_value, dict):
            if "min" in price_value and isinstance(price_value["min"], (int, float, str)) and str(price_value["min"]).replace('.', '', 1).isdigit():
                min_price = float(price_value["min"])
            if "max" in price_value and isinstance(price_value["max"], (int, float, str)) and str(price_value["max"]).replace('.', '', 1).isdigit():
                max_price = float(price_value["max"])
        elif isinstance(price_value, (int, float, str)) and str(price_value).replace('.', '', 1).isdigit():
            max_price = float(price_value) # Wenn nur ein Wert gegeben ist, nehmen wir es als Maximum an (kann später verfeinert werden)
        elif isinstance(price_value, str):
            price_match_range = re.search(r"(\d+)\s*-\s*(\d+)", price_value)
            if price_match_range:
                min_price = float(price_match_range.group(1))
                max_price = float(price_match_range.group(2))
            else:
                price_match_max = re.search(r"bis\s*(\d+)", price_value, re.IGNORECASE)
                if price_match_max:
                    max_price = float(price_match_max.group(1))
                else:
                    price_match_single = re.search(r"(\d+)", price_value)
                    if price_match_single:
                        max_price = float(price_match_single.group(1))

        if min_price is not None or max_price is not None:
            sanitized["price"] = {}
            if min_price is not None:
                sanitized["price"]["min"] = min_price
            if max_price is not None:
                sanitized["price"]["max"] = max_price

    for field in ["city", "country"]:
        if field in filters and isinstance(filters[field], str) and filters[field].strip():
            sanitized[field] = filters[field].strip()
    continents = ["europe", "asia", "africa", "americas", "north america", "south america", "australia", "antarctica"]
    if "country" in sanitized and sanitized["country"].lower() in continents:
        print(f" Kontinent '{sanitized['country']}' als Land erkannt und entfernt.")
        sanitized.pop("country", None)

    for field in ["features", "type", "category", "amenities"]:
        if field in filters and isinstance(filters[field], str):
            sanitized[field] = [item.strip() for item in filters[field].split(',') if item.strip()]
        elif field in filters and isinstance(filters[field], list):
            sanitized[field] = [item.strip() for item in filters[field] if isinstance(item, str) and item.strip()]
        else:
            sanitized[field] = []

    return sanitized
# NLP-Modell laden
nlp = spacy.load("en_core_web_md")

# Load Regions
with open("regional_groups.json", "r", encoding="utf-8") as f:
    REGIONAL_GROUPS = json.load(f)


# Running?
@app.get("/chat")
async def root():
    return {"message": "WhaleChat is here for YOU"}

# Unterkünfte aus Datenbank holen
def get_accommodations():
    conn = sqlite3.connect("accommodations.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM accommodations")
    rows = cursor.fetchall()
    accommodations = [dict(row) for row in rows]
    conn.close()
    return accommodations

# UI (HTML laden)
@app.get("/", response_class=HTMLResponse)
async def chat_ui():
    with open("index.html", "r", encoding="utf-8") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

# Datenmodell für Anfrage
class ChatInput(BaseModel):
    message: str

# 💡 NLP MODUL
def extract_constraints(message):
    doc = nlp(message)
    constraints = {
        "countries": [],
        "min_guests": None,
    }

    for ent in doc.ents:
        if ent.label_ == "GPE":
            constraints["countries"].append(ent.text.lower())

    if "europe" in message.lower():
        constraints["countries"].extend([
            "france", "germany", "spain", "italy", "portugal", "austria", "switzerland", 
            "greece", "croatia", "norway", "sweden", "netherlands", "belgium", "denmark"
        ])

    if "kid" in message.lower() or "child" in message.lower() or "children" in message.lower():
        constraints["min_guests"] = 3

    return constraints

# Hauptlogik: LLM-basierte Filterung und kombinierte Bewertung
@app.post("/")
async def chat_logic(chat_input: ChatInput):
    user_query = chat_input.message

    # 1. LLM-basierte Filter extrahieren
    llm_result = extract_filters(user_query)
    llm_filters = llm_result.get("filters", {})
    sanitized_llm_filters = sanitize_filters(llm_filters)
    summary_text = llm_result.get("summary", "")
    print("\n🎯 LLM-basierte Filter:", sanitized_llm_filters)

    accommodations = get_accommodations()
    llm_filtered_accommodations = accommodations  # Starte mit allen, Filterung ist optional

    # 2. Leichtes Vorfiltern: Nur nach Mindestanzahl an Gästen
    llm_filtered_accommodations = []
    min_guests = sanitized_llm_filters.get("guests", 1)

    for acc in accommodations:
        if acc.get("guests", 0) >= min_guests:
            llm_filtered_accommodations.append(acc)

    print(f"\n✅ Unterkünfte mit mindestens {min_guests} Gästen: {len(llm_filtered_accommodations)}")


        # 3. Bewertung der Unterkünfte
    scored_accommodations = []
    user_vector = nlp(user_query)

    for acc in llm_filtered_accommodations:
        score = 0
        
        # 🔁 JSON-zu-Text-Semantik vorbereiten
        json_text_parts = []
        for key, value in sanitized_llm_filters.items():
            if isinstance(value, list):
                value_str = ", ".join(value)
            elif isinstance(value, dict):
                value_str = f"min: {value.get('min', '')}, max: {value.get('max', '')}"
            else:
                value_str = str(value)
            json_text_parts.append(f"{key.capitalize()}: {value_str}")
        json_description = ". ".join(json_text_parts)
        json_vector = nlp(json_description)

        acc_text = " ".join(str(value).lower() for value in acc.values())
        similarity_score = user_vector.similarity(nlp(acc_text))
        score += similarity_score * 0.1  # leichtes Gewicht
        
        json_similarity_score = json_vector.similarity(nlp(acc_text))
        score += json_similarity_score * 0.15  # JSON-Zu-Text Bewertung


        # Key-Matching mit Gewichtungen
        if sanitized_llm_filters:
            if "country" in sanitized_llm_filters and sanitized_llm_filters["country"].lower() in acc.get("country", "").lower():
                score += 0.5
            if "price" in sanitized_llm_filters and isinstance(sanitized_llm_filters["price"], dict):
                acc_price = acc.get("price", 99999)
                price_filter = sanitized_llm_filters["price"]
                if ("min" not in price_filter or acc_price >= price_filter.get("min", 0)) and \
                   ("max" not in price_filter or acc_price <= price_filter.get("max", 99999)):
                    score += 0.5
            if "guests" in sanitized_llm_filters and acc.get("guests", 0) >= sanitized_llm_filters["guests"]:
                score += 0.3

            for field, weight in [("features", 0.02), ("type", 0.02), ("amenities", 0.02)]:
                for keyword in sanitized_llm_filters.get(field, []):
                    if keyword.lower() in acc_text:
                        score += weight

            for field, weight in [("features", 0.08), ("type", 0.1), ("amenities", 0.09)]:
                for keyword in sanitized_llm_filters.get(field, []):
                    keyword_doc = nlp(keyword.lower())
                    acc_doc = nlp(acc_text.lower())
                    similarity = keyword_doc.similarity(acc_doc)
                    if similarity > 0.7:
                        score += weight
                # 💡 REGION-TO-COUNTRY SCORING
            for region, countries in REGIONAL_GROUPS.items():
                if region.lower() in [t.lower() for t in sanitized_llm_filters.get("type", [])]:
                    acc_country = acc.get("country", "").lower()
                    if acc_country in countries:
                        print(f"🌍 Region Match: {region} matched {acc_country}")
                        score += 0.2



        scored_accommodations.append((acc, score))


      # 4. Sortieren und Auswahl der Top 2
    scored_accommodations.sort(key=lambda item: item[1], reverse=True)
    best_matches = [item[0] for item in scored_accommodations[:2]]

    # The return statement should be aligned properly at the same level as the code above it
    return {
        "results": best_matches,
        "recognized": sanitized_llm_filters,
        "summary": summary_text
    }

