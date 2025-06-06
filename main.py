from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import sqlite3
import spacy
import os
import requests
import json
from dotenv import load_dotenv
import re
from rapidfuzz import process, fuzz
from pathlib import Path

app = FastAPI()

load_dotenv("token.env")

BASE_DIR = Path(__file__).resolve().parent

# Statische Dateien aus flags/ serven
app.mount("/flags", StaticFiles(directory=BASE_DIR / "flags"), name="flags")
app.mount("/icons", StaticFiles(directory="icons"), name="icons")

# Together AI spezifische Variablen
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_MODEL = "mistralai/Mistral-Small-24B-Instruct-2501"  # Ein kostenloses Instruct-Modell (kannst du ändern)
TOGETHER_API_URL = "https://api.together.xyz/v1/completions"
TOGETHER_HEADERS = {
    "Authorization": f"Bearer {TOGETHER_API_KEY}",
    "Content-Type": "application/json"
}

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
    - Only Include a Max Price if limiting words like budget/not expensive etc. or direct price limit are included 
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

Heere is an REal Life Example Output:
    "price": 140,
    "rating": 5,
    "city": "Dubai",
    "amenities": "WiFi, Air conditioning, Beach access, City view, 1 bedroom, kingsize bed",
    "size": 28,
    "category": "Apartment",
    "features": "Gym, Spa, pool, Luxury",
    "availability": "Available",
    "country": "United Arab Emirates",
    "guests": 2,
    "type": "City, Luxury, Modern, Couple, Middle East, Sun, Beach, Skyline"
```

Prompt: {user_prompt}
"""
def print_full_llm_output(response: requests.Response) -> None:
    """Prints the complete LLM API response for debugging purposes."""
    print("\n🔍 FULL LLM API RESPONSE:")
    print("   Status Code:", response.status_code)
    print("   Headers:", response.headers)
    try:
        response_json = response.json()
        print("   Response Body:")
        print(json.dumps(response_json, indent=2))
    except json.JSONDecodeError:
        print("   Response Text (raw):")
        print(response.text)
    print("-" * 50)
    
# JSON wird erstellt und für mich zugreifbar
def extract_filters(user_prompt: str) -> dict:
    full_prompt = build_prompt(user_prompt)

    payload = {
        "model": TOGETHER_MODEL,
        "prompt": full_prompt,
        "max_tokens": 1000,  # Increased from 500
        "temperature": 0.1,  # Lowered from 0.2 for more predictability
        "stop": ["\n```", "```json"],  # Prevents markdown blocks
        "repetition_penalty": 1.2  # Reduces rambling
    }

    print("\n🧠 Anfrage wird gesendet an Together AI...")
    try:
        response = requests.post(TOGETHER_API_URL, headers=TOGETHER_HEADERS, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        
        print_full_llm_output(response)

        result = response.json()
        
        if 'choices' in result and len(result['choices']) > 0 and 'text' in result['choices'][0]:
            result_text = result['choices'][0]['text']
            
            # Clean the text (remove markdown formatting if present)
            cleaned_text = result_text.strip()
            if cleaned_text.startswith('```json') and cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[7:-3].strip()  # Remove ```json markers
            
            # Extract JSON from cleaned text
            json_match = re.search(r'\{.*\}', cleaned_text, re.DOTALL)
            summary_match = re.search(r'"summary"\s*:\s*"([^"]+)"', cleaned_text)

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
  
with open("country_aliases.json", "r", encoding="utf-8") as f:
    COUNTRY_ALIAS_MAP = json.load(f)
  
def load_country_aliases(path="country_aliases.json"):
    with open(path, "r") as f:
        raw = json.load(f)

    alias_map = {}
    for canonical, aliases in raw.items():
        canonical_lower = canonical.lower()
        alias_map[canonical_lower] = canonical_lower
        for alias in aliases:
            alias_map[alias.lower()] = canonical_lower
    return alias_map
    
COUNTRY_ALIAS_MAP = load_country_aliases()
def extract_important_keywords_from_prompt(user_query: str) -> set:
    """Finds Important Keywords that appear verbatim in the user's prompt"""
    user_query_lower = user_query.lower()
    found_keywords = set()
    
    for category_keywords in important_keywords.values():
        for keyword in category_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', user_query_lower):
                found_keywords.add(keyword)
    
    return found_keywords

def normalize_country(name):
    return COUNTRY_ALIAS_MAP.get(name.lower(), name.lower())

def fallback_extraction(user_prompt, filters):
    user_prompt_lower = user_prompt.lower()

    # Load regions and countries
    with open("regional_groups.json", "r", encoding="utf-8") as f:
        REGIONAL_GROUPS = json.load(f)

    with open("travel_terms.json", "r", encoding="utf-8") as f:
        TRAVEL_TERMS = json.load(f)

    with open("cities.json", "r", encoding="utf-8") as f:
        CITY_LIST = json.load(f)

    # Build country and region sets
    all_country_aliases = set(COUNTRY_ALIAS_MAP.keys())
    all_regions = set()

    for region, countries in REGIONAL_GROUPS.items():
        all_regions.add(region.lower())
        for country in countries:
            all_country_aliases.add(country.lower())

    # === Country Detection ===
    country_match, country_score, _ = process.extractOne(
        user_prompt_lower, all_country_aliases, scorer=fuzz.token_set_ratio
    )

    print(f"\n🔍 Fuzzy Country Matching:")
    print(f"   Input: '{user_prompt_lower}'")
    print(f"   Best match: '{country_match}' (Score: {country_score})")

    if country_score > 92:
        normalized_country = normalize_country(country_match)
        filters["country"] = normalized_country
        print(f"   → Set country to: {normalized_country}")

    # === Region Detection ===
    if "country" not in filters:
        region_match, region_score, _ = process.extractOne(
            user_prompt_lower, all_regions, scorer=fuzz.partial_ratio
        )
        if region_score > 85:
            filters.setdefault("type", []).append(region_match.title())
            if region_match in REGIONAL_GROUPS:
                rep_countries = [normalize_country(c) for c in REGIONAL_GROUPS[region_match][:2]]
                filters.setdefault("features", []).extend([f"{region_match} region"] + rep_countries)

    # === City Detection ===
    flat_city_list = {alias.lower(): city for city, aliases in CITY_LIST.items() for alias in aliases}
    city_match, city_score, _ = process.extractOne(
        user_prompt_lower, flat_city_list.keys(), scorer=fuzz.token_set_ratio
    )

    print(f"\n🏙️ Fuzzy City Matching:")
    print(f"   Best match: '{city_match}' (Score: {city_score})")
    if city_score > 92:
        filters["city"] = flat_city_list[city_match].title()
        print(f"   → Set city to: {filters['city']}")

    # === Travel Terms Detection ===
    print(f"\n🧳 Detected Travel Terms:")
    for category, terms in TRAVEL_TERMS.items():
        for term in terms:
            if re.search(r'\b' + re.escape(term) + r'\b', user_prompt_lower):
                if category == "accommodation_types":
                    filters.setdefault("category", []).append(term.title())
                elif category == "travel_styles":
                    filters.setdefault("type", []).append(term.title())
                elif category == "amenities":
                    filters.setdefault("amenities", []).append(term.title())
                elif category == "property_features":
                    filters.setdefault("features", []).append(term.title())
                elif category in ["seasonal_terms", "price_levels"]:
                    filters.setdefault("type", []).append(term.title())
                elif category == "group_types":
                    if term in ["solo", "single"]:
                        filters["guests"] = 1
                    elif term in ["couple", "pair", "romantic"]:
                        filters["guests"] = 2
                    elif term == "family":
                        filters["guests"] = 4
                    elif term in ["friends", "group"]:
                        filters["guests"] = 6
                print(f"   ➤ '{term}' → ({category})")

    # === Price Detection ===
    print(f"\n💰 Price Detection:")
    price_matches = re.finditer(r'(\$|€|£)?\s*(\d{2,4})(?:\s*(?:-|to|bis)\s*(\d{2,4}))?', user_prompt_lower)
    for match in price_matches:
        if match.group(3):  # Price range
            min_price = float(match.group(2))
            max_price = float(match.group(3))
            filters.setdefault("price", {})
            filters["price"]["min"] = min_price
            filters["price"]["max"] = max_price
            print(f"   → Detected price range: {min_price} to {max_price}")
        else:
            price = float(match.group(2))
            filters.setdefault("price", {})
            filters["price"]["max"] = price
            print(f"   → Detected max price: {price}")

    # === Guests Detection ===
    if "guests" in filters:
        print(f"\n👥 Guest Count: {filters['guests']}")

    return filters

    
# NLP-Modell laden
nlp = spacy.load("en_core_web_md")

# Load Regions
with open("regional_groups.json", "r", encoding="utf-8") as f:
    REGIONAL_GROUPS = json.load(f)
with open("important_keywords.json", "r", encoding="utf-8") as f:
    important_keywords = json.load(f)

def calculate_max_possible_score(sanitized_llm_filters):
    max_score = 0
    
    # Base similarity scores (user query + json description + description similarity)
    max_score += 0.45 + 0.45 + 0.7  # = 1.6
    
    # Country/region points
    if "country" in sanitized_llm_filters and sanitized_llm_filters["country"]:
        max_score += 1.0  # country match
        max_score += 0.75 # potential price match
        max_score += 0.5  # potential guests match
        max_score += 0.75 # potential city match
    elif any(region.lower() in REGIONAL_GROUPS 
             for region in sanitized_llm_filters.get("type", [])):
        max_score += 1.0  # region match
    
    # Exact keyword matches (features, type, amenities)
    keyword_fields = ["features", "type", "amenities"]
    max_keyword_points = sum(
        len(sanitized_llm_filters.get(field, [])) * 0.3 
        for field in keyword_fields
    )
    max_score += min(max_keyword_points, 3.0)  # Cap at 3.0
    
    # Important keywords bonus
    important_keyword_count = sum(
        1 for field in keyword_fields
        for keyword in sanitized_llm_filters.get(field, [])
        if any(keyword.lower() in keywords 
               for keywords in important_keywords.values())
    )
    max_score += important_keyword_count * 0.2
    
    return max_score

# Running?
@app.get("/chat")
async def root():
    return {"message": "WhaleChat is here for YOU"}
    
@app.get("/accommodations", response_class=JSONResponse)
async def get_all_accommodations():
    """
    Gibt alle Unterkünfte aus der Datenbank als JSON zurück
    """
    accommodations = get_accommodations()  # Nutzt die existierende Funktion
    return accommodations
@app.get("/recommendations")

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
    
    #Print the user's original prompt with timestamp
    print("\n" + "="*50)
    print("💬 USER PROMPT:", user_query)
    print("="*50 + "\n")
    
    
    # 1. LLM-basierte Filter extrahieren
    llm_result = extract_filters(user_query)
    llm_filters = llm_result.get("filters", {})
    sanitized_llm_filters = sanitize_filters(llm_filters)
    summary_text = llm_result.get("summary", "")
    print("\n🎯 LLM-Key-Words:", sanitized_llm_filters)
    
    # Check if filters are too weak
    weak_filters = (
        not sanitized_llm_filters.get("country") and 
        len(sanitized_llm_filters.get("features", [])) + 
        len(sanitized_llm_filters.get("type", [])) < 1
    )
    
    if weak_filters:
        print("⚡ Detected weak LLM output → Activating fallback extraction...")
        sanitized_llm_filters = fallback_extraction(user_query, sanitized_llm_filters)
        
        # Check if fallback also produced weak results
        fallback_weak = (
            not sanitized_llm_filters.get("country") and 
            len(sanitized_llm_filters.get("features", [])) + 
            len(sanitized_llm_filters.get("type", [])) < 1
        )
        
        if fallback_weak:
            print("⚠️ Both LLM and fallback extraction failed → Using pure similarity matching")
            return pure_similarity_matching(user_query)
    
    prompt_keywords = extract_important_keywords_from_prompt(user_query)
    print(f"🔑 Important Keywords im Prompt: {prompt_keywords}") 
    
    accommodations = get_accommodations()
    llm_filtered_accommodations = accommodations  # Kein harter Gäste-Filter
    min_guests = sanitized_llm_filters.get("guests", 1)  # Wird nur noch zum Scoren genutzt

    # Max-Possible Score idk if needed
    max_possible_score = calculate_max_possible_score(sanitized_llm_filters)
    
    scored_accommodations = []
    user_vector = nlp(user_query)
    matched_keywords = set()  # Wird jetzt nur für die Top 2 verwendet (GEAENDERT)
    for acc in llm_filtered_accommodations:
        score = 0
        
        #Similarity calculations
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

        #Prompt zu Haus Text
        acc_text = " ".join(str(value).lower() for value in acc.values())
        similarity_score = user_vector.similarity(nlp(acc_text))
        score += similarity_score * 0.25
        
        #LLM Output zu Haus Text
        json_similarity_score = json_vector.similarity(nlp(acc_text))
        score += json_similarity_score * 0.25

        # Description similarity
        description = str(acc.get("description")) or ""
        acc_description = description.lower()
        if acc_description:
            description_similarity = user_vector.similarity(nlp(acc_description))
            score += description_similarity * 0.5

        #Country + Price + Guest Points
        if sanitized_llm_filters:
            if "country" in sanitized_llm_filters:
                user_country = normalize_country(sanitized_llm_filters["country"])
                acc_country = normalize_country(acc.get("country", ""))
                if user_country == acc_country:
                    score += 1.5
                    if "price" in sanitized_llm_filters and isinstance(sanitized_llm_filters["price"], dict):
                        acc_prices = acc.get("price", [])
                        if not isinstance(acc_prices, list):
                            acc_prices = [acc_prices]  # falls es nur ein einzelner Preis ist

                        price_filter = sanitized_llm_filters["price"]
                        min_price = price_filter.get("min", 0)
                        max_price = price_filter.get("max", 99999)

                        # Prüfe, ob *ein beliebiger* Preis im Bereich liegt
                        if any(min_price <= float(price) <= max_price for price in acc_prices if str(price).replace('.', '', 1).isdigit()):
                            score += 1.5
                    if "guests" in sanitized_llm_filters:
                        acc_guests = str(acc.get("guests", "0")).split(",")  # Split multiple values
                        acc_guests = [int(g.strip()) for g in acc_guests if g.strip().isdigit()]  # Convert to integers
                        required_guests = sanitized_llm_filters["guests"]
                        
                        if acc_guests and any(g >= required_guests for g in acc_guests):
                            score += 0.75

                        # Prüfe, ob *mindestens ein Zimmer* genügend Gäste aufnehmen kann
                        if any(room_capacity >= required_guests for room_capacity in acc_guests):
                            score += 0.75
                    if "city" in sanitized_llm_filters and acc.get("city", 0) == sanitized_llm_filters["city"]:
                        score += 2
        
            # 1. Alle Keywords sammeln (ohne Duplikate)
            all_keywords = set()
            for field in ["features", "type", "amenities"]:
                for keyword in sanitized_llm_filters.get(field, []):
                    all_keywords.add(keyword.lower())

            # 2. Jedes Keyword nur einmal bewerten
            for keyword in all_keywords:
                is_important = any(
                    keyword in category_keywords 
                    for category_keywords in important_keywords.values()
                )
                weight = 0.2 if is_important else 0.1
                
                # Prüfen wo das Keyword erscheint
                in_main_text = keyword in acc_text
                in_description = keyword in str(acc.get("description", "")).lower()
                
                if in_main_text or in_description:
                    # Bonus für direkte Prompt-Matches
                    if keyword in prompt_keywords:
                        weight *= 1.5
                        print(f"   💥 DIRECT PROMPT MATCH: '{keyword}' → +{round(weight, 2)}")
                    
                    score += weight
                    matched_keywords.add(keyword)
                    
                    # Debug-Info
                    status = "(direkt im Text)" if in_main_text else "(in Beschreibung)"
                    importance = "🌟 WICHTIG" if is_important else "   normal"
                    print(f"      - {importance}: '{keyword}' {status} → +{round(weight, 2)}")

            if "country" not in sanitized_llm_filters or not sanitized_llm_filters["country"]:
                regions_in_type = [t.lower() for t in sanitized_llm_filters.get("type", [])]
                matched_regions = set()
                
                for region, countries in REGIONAL_GROUPS.items():
                    if region.lower() in regions_in_type:
                        acc_country = acc.get("country", "").lower()
                        if acc_country in countries and region not in matched_regions:
                            print(f"🌍 Region Match: {region} matched {acc_country} (+1.0)")
                            score += 1
                            matched_regions.add(region)
                       
                    
                    
        scored_accommodations.append((acc, score))

    scored_accommodations.sort(key=lambda item: item[1], reverse=True)
    best_matches = [item[0] for item in scored_accommodations[:2]]

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

        #PRINTEN DER TOP 2 SCORES:
        print(f"\n🏆 Top-{idx} Accommodation: {acc.get('name', 'N/A')} (ID: {acc.get('id', 'N/A')})")
        print(f"   Gesamtpunktzahl: {round(score, 2)} ({round((score / max_possible_score) * 100, 1)}% von Max)")
                
        user_sim = user_vector.similarity(nlp(acc_text)) * 0.5
        json_sim = json_vector.similarity(nlp(acc_text)) * 0.5
        print(f"   🔍 Ähnlichkeits-Punkte:")
        print(f"      - User-Query: {round(user_sim, 2)} (Raw: {round(user_vector.similarity(nlp(acc_text)), 2)})")
        print(f"      - JSON-Filter: {round(json_sim, 2)} (Raw: {round(json_vector.similarity(nlp(acc_text)), 2)})")
                
        if acc_description:
            desc_sim = user_vector.similarity(nlp(acc_description)) * 0.8
            print(f"      - Description: {round(desc_sim, 2)} (Raw: {round(user_vector.similarity(nlp(acc_description)), 2)})")

        print("   ✅ Filter-Matches:")
        if "country" in sanitized_llm_filters and sanitized_llm_filters["country"].lower() in acc.get("country", "").lower():
            print(f"      - Land: {sanitized_llm_filters['country']} (+1.0)")
                
        if "price" in sanitized_llm_filters and isinstance(sanitized_llm_filters["price"], dict):
            acc_prices = str(acc.get("price", "99999")).split(",")  # Split multiple values
            acc_prices = [float(p.strip()) for p in acc_prices if p.strip().replace('.', '', 1).isdigit()]  # Convert to floats
            price_filter = sanitized_llm_filters["price"]
            min_price = price_filter.get("min", 0)
            max_price = price_filter.get("max", 99999)
            
            if any((min_price <= price <= max_price) for price in acc_prices):
                print(f"      - Preisrange: {price_filter} (+0.75)")
                
        if "guests" in sanitized_llm_filters:
            acc_guests = str(acc.get("guests", "0")).split(",")  # Split multiple values
            acc_guests = [int(g.strip()) for g in acc_guests if g.strip().isdigit()]  # Convert to integers
            if acc_guests and any(g >= sanitized_llm_filters["guests"] for g in acc_guests):
                print(f"      - Gästezahl: {acc.get('guests')} (≥ {sanitized_llm_filters['guests']}) (+0.5)")
        if "city" in sanitized_llm_filters and acc.get("city", "").lower() == sanitized_llm_filters["city"].lower():
            print(f"      - City: {sanitized_llm_filters['city']} (+0.75)")

        # NEU: Keyword-Analyse für Top 2
        print("   🔍 Alle Keywords:")
        for field in ["features", "type", "amenities"]:
            for keyword in sanitized_llm_filters.get(field, []):
                keyword_lower = keyword.lower()
                is_important = any(
                    keyword_lower in category_keywords 
                    for category_keywords in important_keywords.values()
                )
                
                in_main_text = keyword_lower in acc_text
                in_description = keyword_lower in str(acc.get("description", "")).lower()
                
                if in_main_text or in_description:
                    status = "(direkt im Text)" if in_main_text else "(in Beschreibung)"
                    importance = "🌟 WICHTIG" if is_important else "   normal"
                    print(f"      - {importance}: '{keyword}' {status}")

    return {
        "results": best_matches,
        "recognized": sanitized_llm_filters,
        "summary": summary_text
    }

def pure_similarity_matching(user_query: str) -> dict:
    """Vergleicht die Beschreibung mit der Nutzeranfrage für Similarity Matching"""
    accommodations = get_accommodations()
    user_doc = nlp(user_query.lower())

    # Score each accommodation based on prompt similarity with description
    scored = []
    for acc in accommodations:
        description = str(acc.get("description", "")).lower()
        if not description:
            continue

        acc_doc = nlp(description)
        similarity = acc_doc.similarity(user_doc)  # Hier ist die Änderung!
        scored.append((acc, similarity))

    # Sort by similarity (descending) and get top 2
    scored.sort(key=lambda x: x[1], reverse=True)
    best_matches = [item[0] for item in scored[:2]]

    # Print debug info
    for idx, (acc, score) in enumerate(scored[:2], 1):
        print(f"\n🏆 Top-{idx} Similarity Match (Reversed): {acc.get('name', 'N/A')}")
        print(f"    Similarity Score: {round(score, 2)}")
        print(f"    Description: {acc.get('description', 'N/A')[:200]}...")

    return {
        "results": best_matches,
        "recognized": {"method": "pure_similarity_reversed"},
        "summary": "Please be patient and try again, as our AI algorithm had a problem and we need to use the simplest one..."
    }
