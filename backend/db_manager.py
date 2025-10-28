# /backend/db_manager.py
import sqlite3
from datetime import datetime
import os

DATABASE_FILE = os.path.join(os.path.dirname(__file__), 'anpr_log.db')

def setup_database():
    """Initializes the SQLite database."""
    try:
        conn = sqlite3.connect(DATABASE_FILE)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY,
                timestamp TEXT,
                plate_number TEXT,
                confidence REAL,
                image_filename TEXT
            )
        """)
        conn.commit()
        conn.close()
        print(f"DB Manager: Database initialized at {DATABASE_FILE}")
    except Exception as e:
        print(f"DB Manager Error during setup: {e}")

def log_detection(plate_number: str, confidence: float, filename: str):
    """Logs a detection event."""
    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    c.execute("""
        INSERT INTO detections (timestamp, plate_number, confidence, image_filename)
        VALUES (?, ?, ?, ?)
    """, (current_time, plate_number, confidence, filename))
    
    conn.commit()
    conn.close()
    
def get_detection_history():
    """Fetches all logged plate detection records from the database."""
    conn = sqlite3.connect(DATABASE_FILE)
    c = conn.cursor()
    
    # Select all records, ordering by timestamp (most recent first)
    c.execute("SELECT id, timestamp, plate_number, confidence FROM detections ORDER BY timestamp DESC")
    
    # Fetch results and convert to a list of dictionaries for JSON output
    columns = [col[0] for col in c.description]
    history = [dict(zip(columns, row)) for row in c.fetchall()]
    
    conn.close()
    return history

# Initialize on import
setup_database()