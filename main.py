from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import sqlite3
import spacy
import random

app = FastAPI()
templates = Jinja2Templates(directory=".")

# spaCy NLP-Modell laden
nlp = spacy.load("en_core_web_md")

# Root Endpoint
@app.get("/")
async def root():
    return {"message": "WhaleChat is here for YOU"}

# Datenbank-Funktion
def get_accommodations():
    conn = sqlite3.connect("accommodations.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM accommodations")
    rows = cursor.fetchall()
    accommodations = [dict(row) for row in rows]
    conn.close()
    return accommodations

# Alle Unterkünfte anzeigen
@app.get("/accommodations")
async def read_accommodations():
    accommodations = get_accommodations()
    return JSONResponse(content=accommodations)

# Chat UI laden
@app.get("/whalechat", response_class=HTMLResponse)
async def chat_ui(request: Request):
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# Nutzereingabe-Modell
class ChatInput(BaseModel):
    message: str

# Semantische Suche mit spaCy
@app.post("/whalechat")
async def chat_logic(chat_input: ChatInput):
    user_query = chat_input.message.lower()
    user_doc = nlp(user_query)

    accommodations = get_accommodations()
    scored_matches = []

    for acc in accommodations:
        acc_text = f"{acc.get('title', '')} {acc.get('description', '')} {acc.get('location', '')} {acc.get('tags', '')}".lower()
        acc_doc = nlp(acc_text)
        similarity = user_doc.similarity(acc_doc)
        scored_matches.append((similarity, acc))

    scored_matches.sort(reverse=True, key=lambda x: x[0])
    best = [acc for score, acc in scored_matches if score > 0.6][:2]

    return {"results": best}
