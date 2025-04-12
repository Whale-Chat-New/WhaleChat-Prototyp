from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import sqlite3
import random
import re

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# 1. Keep original root endpoint
@app.get("/")
async def root():
    return {"message": "WhaleChat is here for YOU"}

# 2. Keep original database function
def get_accommodations():
    conn = sqlite3.connect("accommodations.db")
    conn.row_factory = sqlite3.Row  # 👈 This line is key
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM accommodations")
    rows = cursor.fetchall()
    accommodations = []
    for row in rows:
        accommodation = dict(row)  # 👈 Automatically creates a dict with column names
        accommodations.append(accommodation)
    conn.close()
    return accommodations


# 3. Keep original /accommodations endpoint
@app.get("/accommodations")
async def read_accommodations():
    accommodations = get_accommodations()
    return JSONResponse(content=accommodations)

# 4. Keep original chat UI endpoint
@app.get("/whalechat", response_class=HTMLResponse)
async def chat_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 5. Keep original ChatInput model
class ChatInput(BaseModel):
    message: str

# 6. NEW IMPROVED SEARCH LOGIC (only this part changed)
@app.post("/whalechat")
async def chat_logic(chat_input: ChatInput):
    # Extract words from input
    search_terms = re.findall(r'\w+', chat_input.message.lower())
    
    if not search_terms:
        return {"results": []}
    
    accommodations = get_accommodations()
    matches = []
    
    for acc in accommodations:
        # Combine ALL field values into one searchable string
        search_text = " ".join(str(value).lower() for value in acc.values())
        
        # Check if ALL terms appear anywhere in ANY field
        if all(term in search_text for term in search_terms):
            matches.append(acc)
    
    # Return 2 random results if more than 2 matches
    if len(matches) > 2:
        return {"results": random.sample(matches, 2)}
    return {"results": matches}