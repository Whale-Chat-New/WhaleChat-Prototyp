from fastapi import FastAPI
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import sqlite3
import spacy

app = FastAPI()

# NLP-Modell laden
nlp = spacy.load("en_core_web_md")

# Root
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

# 💡 Extrahiere Ankerpunkte aus Text
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

# Hauptlogik: semantische Suche mit Regeln
@app.post("/")
async def chat_logic(chat_input: ChatInput):
    user_query = chat_input.message
    user_vector = nlp(user_query)
    constraints = extract_constraints(user_query)

    accommodations = get_accommodations()
    filtered = []

    for acc in accommodations:
        acc_text = " ".join(str(value).lower() for value in acc.values())

        if constraints["countries"]:
            if not any(country in acc_text for country in constraints["countries"]):
                continue

        if constraints["min_guests"]:
            if "guests" in acc and int(acc["guests"]) < constraints["min_guests"]:
                continue

        filtered.append(acc)

    matches = []
    for acc in filtered:
        acc_vector = nlp(" ".join(str(v).lower() for v in acc.values()))
        sim_score = user_vector.similarity(acc_vector)
        matches.append((acc, sim_score))

    matches.sort(key=lambda x: x[1], reverse=True)
    best_matches = [match[0] for match in matches[:2]]

    return {"results": best_matches}
