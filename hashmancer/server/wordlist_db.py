import sqlite3
from typing import Generator

from .app.config import WORDLIST_DB_PATH

DB_PATH = WORDLIST_DB_PATH
DB_PATH.parent.mkdir(parents=True, exist_ok=True)


def connect() -> sqlite3.Connection:
    """Return a connection to the wordlist database."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "CREATE TABLE IF NOT EXISTS wordlists("
        "id INTEGER PRIMARY KEY, name TEXT UNIQUE, data BLOB)"
    )
    return conn


def begin_insert(name: str) -> sqlite3.Connection:
    """Create or replace a wordlist row and return an open connection."""
    conn = connect()
    conn.execute("BEGIN")
    conn.execute("DELETE FROM wordlists WHERE name=?", (name,))
    conn.execute("INSERT INTO wordlists(name, data) VALUES(?, zeroblob(0))", (name,))
    return conn


def append_chunk(conn: sqlite3.Connection, name: str, chunk: bytes) -> None:
    conn.execute("UPDATE wordlists SET data = data || ? WHERE name=?", (sqlite3.Binary(chunk), name))


def finish_insert(conn: sqlite3.Connection) -> None:
    conn.commit()
    conn.close()


def stream_wordlist(name: str, chunk_size: int = 8192) -> Generator[bytes, None, None]:
    """Yield a wordlist's data in chunks."""
    conn = connect()
    row = conn.execute(
        "SELECT id, length(data) FROM wordlists WHERE name=?", (name,)
    ).fetchone()
    if not row:
        conn.close()
        return
    rowid, length = row
    blob = conn.blobopen("wordlists", "data", rowid, readonly=True)
    offset = 0
    try:
        while offset < length:
            chunk = blob.read(min(chunk_size, length - offset))
            if not chunk:
                break
            offset += len(chunk)
            yield chunk
    finally:
        blob.close()
        conn.close()


def list_names() -> list[str]:
    conn = connect()
    rows = conn.execute("SELECT name FROM wordlists").fetchall()
    conn.close()
    return [r[0] for r in rows]
