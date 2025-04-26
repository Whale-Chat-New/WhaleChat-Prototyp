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

Read between the lines and get as MUCH out of the prompt as you can. Assume what the person writing the prompt would like to have, but is not saying.

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
    - Include a smart `summary` at the end like:
        `"summary": "These accommodations have a pool and are close to the beach, perfect for creating your next travel memory."`
5. Set at least 5 Words as Key Words in `type`
    - At least one about the: Person, Travel Style, Sourroundng, Budget, Location
    
    
**Final Format Example:**

(Do never change structure, just fill in with user’s intent)

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
        sanitized["guests"] = 1  # Standardwert

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
        if "price" in sanitized and sanitized["price"].get("max", 99999) == 0.0:
            print("⚡ Detected price max=0.0 → removing price filter (interpreted as 'no limit')")
            sanitized.pop("price")


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

def calculate_max_possible_score(sanitized_llm_filters):
    max_score = 0

    # Base similarity scores
    max_score += 0.75  # user query similarity
    max_score += 0.75  # json description similarity

    # Fixed bonuses
    # Land oder Region Punkte, aber niemals doppelt
    if "country" in sanitized_llm_filters and sanitized_llm_filters["country"]:
        # Country vorhanden (z.B. France, Italy)
        max_score += 1  # ordentlich Punkte für echtes Land
    elif "type" in sanitized_llm_filters:
        for region in sanitized_llm_filters["type"]:
            if region.lower() in REGIONAL_GROUPS:
                max_score += 1  # Region (Kontinent) bekommt auch 1 Punkt
                break  # Nur 1x addieren, auch wenn mehrere Regionen drinstehen

    if "price" in sanitized_llm_filters:
        max_score += 0.75
    
    if "guests" in sanitized_llm_filters:
        max_score += 0.5

    # Exact keyword matches
    for field in ["features", "type", "amenities"]:
        if field in sanitized_llm_filters:
            max_score += len(sanitized_llm_filters[field]) * 0.05  # 0.05 pro Keyword/Feld

    # Semantic matches (gesamthaft begrenzt auf 0.8)
    max_score += 0.8

    # Region matching (optional)
    if "type" in sanitized_llm_filters:
        for region in sanitized_llm_filters["type"]:
            if region.lower() in REGIONAL_GROUPS:
                max_score += 0.2

    return max_score

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
    llm_filtered_accommodations = accommodations  # Kein harter Gäste-Filter
    min_guests = sanitized_llm_filters.get("guests", 1)  # Wird nur noch zum Scoren genutzt

    # Calculate dynamic max possible score
    max_possible_score = calculate_max_possible_score(sanitized_llm_filters)
    
    scored_accommodations = []
    user_vector = nlp(user_query)
    matched_keywords = set()
    for acc in llm_filtered_accommodations:
        score = 0
        
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
        score += similarity_score * 0.8
        
        json_similarity_score = json_vector.similarity(nlp(acc_text))
        score += json_similarity_score * 0.8

        if sanitized_llm_filters:
            if "country" in sanitized_llm_filters and sanitized_llm_filters["country"].lower() in acc.get("country", "").lower():
                score += 1
            if "price" in sanitized_llm_filters and isinstance(sanitized_llm_filters["price"], dict):
                acc_price = acc.get("price", 99999)
                price_filter = sanitized_llm_filters["price"]
                if ("min" not in price_filter or acc_price >= price_filter.get("min", 0)) and \
                   ("max" not in price_filter or acc_price <= price_filter.get("max", 99999)):
                    score += 0.75
            if "guests" in sanitized_llm_filters and acc.get("guests", 0) >= sanitized_llm_filters["guests"]:
                score += 0.5

            
                    
            #Exakte Keywords
            for field, weight in [("features", 0.05), ("type", 0.05), ("amenities", 0.05)]:
                for keyword in sanitized_llm_filters.get(field, []):
                    if keyword.lower() in acc_text:
                        score += weight
                        matched_keywords.add(keyword.lower())  # <--- HINZUFÜGEN


            # Region matches NUR wenn kein Land angegeben
            if "country" not in sanitized_llm_filters or not sanitized_llm_filters["country"]:
                for region, countries in REGIONAL_GROUPS.items():
                    if region.lower() in [t.lower() for t in sanitized_llm_filters.get("type", [])]:
                        acc_country = acc.get("country", "").lower()
                        if acc_country in countries:
                            print(f"🌍 Region Match: {region} matched {acc_country}")
                            score += 1
                       
            # Nur wenn die Unterkunft schon mindestens 50% des maximalen Scores erreicht hat → semantisches Matching zulassen
            if score >= (0.5 * max_possible_score):
                semantic_score, debug = score_semantic_keywords(
                    sanitized_llm_filters, 
                    acc, 
                    nlp,
                    matched_keywords=matched_keywords,  # <--- HINZUGEFÜGT
                    semantic_weights_flat=0.1,
                    max_semantic_points=0.8
                )

                score += semantic_score
                for line in debug:
                    print(line)
                    
                    
        scored_accommodations.append((acc, score))

    scored_accommodations.sort(key=lambda item: item[1], reverse=True)
    best_matches = [item[0] for item in scored_accommodations[:2]]

    print("\n🔢 DETAILED SCORING BREAKDOWN:")
    print(f"Maximal mögliche Punktzahl (dynamisch berechnet): {round(max_possible_score, 2)}")
    
    for idx, (acc, score) in enumerate(scored_accommodations[:2], 1):
        acc_text = " ".join(str(value).lower() for value in acc.values())
        user_vector = nlp(user_query)
        
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

        print(f"\n🏆 Top-{idx} Accommodation: {acc.get('name', 'N/A')} (ID: {acc.get('id', 'N/A')})")
        print(f"   Gesamtpunktzahl: {round(score, 2)} ({round((score / max_possible_score) * 100, 1)}% von Max)")
        
        user_sim = user_vector.similarity(nlp(acc_text)) * 0.75
        json_sim = json_vector.similarity(nlp(acc_text)) * 0.75
        print(f"   🔍 Ähnlichkeits-Punkte:")
        print(f"      - User-Query: {round(user_sim, 2)} (Raw: {round(user_vector.similarity(nlp(acc_text)), 2)})")
        print(f"      - JSON-Filter: {round(json_sim, 2)} (Raw: {round(json_vector.similarity(nlp(acc_text)), 2)})")
        
        print("   ✅ Filter-Matches:")
        if "country" in sanitized_llm_filters and sanitized_llm_filters["country"].lower() in acc.get("country", "").lower():
            print(f"      - Land: {sanitized_llm_filters['country']} (+1)")
        
        if "price" in sanitized_llm_filters and isinstance(sanitized_llm_filters["price"], dict):
            acc_price = acc.get("price", 99999)
            price_filter = sanitized_llm_filters["price"]
            if ("min" not in price_filter or acc_price >= price_filter.get("min", 0)) and \
               ("max" not in price_filter or acc_price <= price_filter.get("max", 99999)):
                print(f"      - Preisrange: {price_filter} (+0.75)")
        
        if "guests" in sanitized_llm_filters and acc.get("guests", 0) >= sanitized_llm_filters["guests"]:
            print(f"      - Gästezahl: {acc.get('guests')} (≥ {sanitized_llm_filters['guests']}) (+0.5)")
        
        print("   🔑 Keyword-Matches:")
        for field in ["features", "type", "amenities"]:
            for keyword in sanitized_llm_filters.get(field, []):
                if keyword.lower() in acc_text:
                    print(f"      - Exact Match: {field}='{keyword}' (+0.05)")

    return {
        "results": best_matches,
        "recognized": sanitized_llm_filters,
        "summary": summary_text
    }
    
def score_semantic_keywords(sanitized_llm_filters, acc, nlp, matched_keywords=None, semantic_weights_flat=0.1, max_semantic_points=0.8):
    if matched_keywords is None:
        matched_keywords = set()

    scored_keywords = set()
    semantic_score = 0.0
    debug_output = []

    all_keywords = set()
    for field in ["features", "type", "amenities"]:
        all_keywords.update([kw.strip().lower() for kw in sanitized_llm_filters.get(field, [])])

    for keyword in all_keywords:
        if keyword in scored_keywords or keyword in matched_keywords:
            continue

        keyword_doc = nlp(keyword)
        best_similarity = 0.0
        best_match_item = ""
        best_field = ""

        for acc_field in ["features", "type", "amenities"]:
            acc_values = acc.get(acc_field, [])
            if isinstance(acc_values, str):
                acc_values = [x.strip() for x in acc_values.split(',')]
            elif not isinstance(acc_values, list):
                continue

            for acc_item in acc_values:
                acc_item_lower = acc_item.strip().lower()
                if keyword == acc_item_lower:
                    continue  # already handled as exact match

                acc_doc = nlp(acc_item_lower)
                if keyword_doc.vector_norm and acc_doc.vector_norm:
                    similarity = keyword_doc.similarity(acc_doc)
                else:
                    similarity = 0.0

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_item = acc_item
                    best_field = acc_field

        if best_similarity >= 0.8:
            semantic_score += semantic_weights_flat
            scored_keywords.add(keyword)
            debug_output.append(
                f"   🔑 Semantic-Match: '{keyword}' ≈ '{best_match_item}' in {best_field} (Sim: {round(best_similarity, 2)}) → +{semantic_weights_flat}"
            )
        elif best_similarity >= 0.7:
            weighted = round(semantic_weights_flat * 0.7, 3)
            semantic_score += weighted
            scored_keywords.add(keyword)
            debug_output.append(
                f"   🔑 Semantic-Match: '{keyword}' ≈ '{best_match_item}' in {best_field} (Sim: {round(best_similarity, 2)}) → +{weighted}"
            )

        if semantic_score >= max_semantic_points:
            break

    return semantic_score, debug_output
