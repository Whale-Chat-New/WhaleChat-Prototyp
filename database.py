import sqlite3

connection = sqlite3.connect("accommodations.db")
cursor = connection.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS accommodations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    price REAL NOT NULL,
    rating REAL,
    location TEXT NOT NULL,
    image_url TEXT,
    amenities TEXT,
    size REAL,
    category TEXT,
    features TEXT,
    availability TEXT
)
''')

connection.commit()
connection.close()
