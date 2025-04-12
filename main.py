from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
import sqlite3
import random
import re
import os

app = FastAPI()

# 1. Root endpoint
@app.get("/")
async def root():
    return {"message": "WhaleChat is here for YOU"}

# 2. Database logic
def get_accommodations():
    conn = sqlite3.connect("accommodations.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM accommodations")
    rows = cursor.fetchall()
    accommodations = [dict(row) for row in rows]
    conn.close()
    return accommodations

# 3. All accommodations
@app.get("/accommodations")
async def read_accommodations():
    return JSONResponse(content=get_accommodations())

# 4. Chat UI – direktes Einlesen von HTML
@app.get("/whalechat", response_class=HTMLResponse)
async def chat_ui():
    with open("index.html", "r", encoding="utf-8") as file:
        html_content = file.read()
    return HTMLResponse(content=html_content)

# 5. ChatInput Model
class ChatInput(BaseModel):
    message: str

# 6. Chat logic
@app.post("/whalechat")
async def chat_logic(chat_input: ChatInput):
    search_terms = re.findall(r'\w+', chat_input.message.lower())
    if not search_terms:
        return {"results": []}
    
    accommodations = get_accommodations()
    matches = []

    for acc in accommodations:
        search_text = " ".join(str(value).lower() for value in acc.values())
        if all(term in search_text for term in search_terms):
            matches.append(acc)

    return {"results": random.sample(matches, 2)} if len(matches) > 2 else {"results": matches}
