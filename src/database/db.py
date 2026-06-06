import sqlite3
import uuid
from pathlib import Path

ROOT    = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "users.db"

def get_connection():
    # ensure directory exists (Docker build may exclude data/)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row  # return dict-like rows
    return conn

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    # users table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id    INTEGER PRIMARY KEY AUTOINCREMENT,
            public_id  TEXT UNIQUE,
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

    cursor.execute("PRAGMA table_info(users)")
    columns = [row[1] for row in cursor.fetchall()]
    if "public_id" not in columns:
        cursor.execute("ALTER TABLE users ADD COLUMN public_id TEXT")
        cursor.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_users_public_id ON users(public_id)")
        cursor.execute("SELECT user_id FROM users WHERE public_id IS NULL")
        rows = cursor.fetchall()
        for row in rows:
            cursor.execute(
                "UPDATE users SET public_id = ? WHERE user_id = ?",
                (str(uuid.uuid4()), row[0])
            )

    conn.commit()
    conn.close()
    print("Database initialised!")

if __name__ == "__main__":
    init_db()