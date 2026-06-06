import sqlite3
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "users.db"

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # return dict-like rows
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # user preferences from onboarding
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_preferences (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id    INTEGER,
            anime_id   INTEGER,
            rating     INTEGER,
            source     TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)

    # genre preferences
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS user_genres (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id  INTEGER,
            genre    TEXT,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)

    conn.commit()
    conn.close()
    print("Database initialised!")

if __name__ == "__main__":
    init_db()