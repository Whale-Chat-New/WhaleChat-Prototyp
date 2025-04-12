from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import sqlite3
import random
import spacy  # Import spaCy für NLP

app = FastAPI()

# Lade das spaCy-Modell
nlp = spacy.load("en_core_web_md")  # Mittelgroßes Modell für Semantik

# 1. Root endpoint
@app.get("/")
async def root():
    return {"message": "WhaleChat is here for YOU"}

# 2. Hole Unterkünfte aus der Datenbank
def get_accommodations():
    conn = sqlite3.connect("accommodations.db")
    conn.row_factory = sqlite3.Row  # Rückgabe als Diktate
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM accommodations")
    rows = cursor.fetchall()
    accommodations = [dict(row) for row in rows]
    conn.close()
    return accommodations

# 3. Alle Unterkünfte zurückgeben
@app.get("/accommodations")
async def read_accommodations():
    return JSONResponse(content=get_accommodations())

# 4. Chat UI (Unverändert)
@app.get("/whalechat", response_class=HTMLResponse)
async def chat_ui():
    with open("index.html", "r", encoding="utf-8") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

# 5. ChatInput Modell
class ChatInput(BaseModel):
    message: str

# 6. Neue Logik für die semantische Suche
@app.post("/whalechat")
async def chat_logic(chat_input: ChatInput):
    # Benutzeranfrage verarbeiten
    user_query = chat_input.message
    user_vector = nlp(user_query)  # Anfrage in Vektor umwandeln

    # Hole alle Unterkünfte aus der Datenbank
    accommodations = get_accommodations()
    matches = []

    for acc in accommodations:
        # Kombiniere die Textfelder der Unterkunft zu einem einzigen String
        accommodation_text = " ".join(str(value).lower() for value in acc.values())
        acc_vector = nlp(accommodation_text)  # Unterkunftstext in Vektor umwandeln

        # Berechne die Ähnlichkeit zwischen der Anfrage und der Unterkunft
        similarity_score = user_vector.similarity(acc_vector)

        # Füge das Ergebnis mit der Ähnlichkeit zum Treffer-Array hinzu
        matches.append((acc, similarity_score))

    # Sortiere nach der Ähnlichkeit (höchste zuerst)
    matches.sort(key=lambda x: x[1], reverse=True)

    # Gib die Top 2 besten Treffer zurück
    best_matches = [match[0] for match in matches[:2]]
    
    return {"results": best_matches}
