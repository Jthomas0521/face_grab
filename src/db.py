import faiss
import numpy as np
import sqlite3
import os
import logging

logging.basicConfig(level=logging.INFO)

DB_PATH = "face_metadata.db"
INDEX_PATH = "face_index.faiss"
DIM = 128  # Facenet

# Init index if not present
if not os.path.exists(INDEX_PATH):
    logging.info(f"FAISS index not found at {INDEX_PATH}. Initializing new index.")
    faiss.write_index(faiss.IndexFlatL2(DIM), INDEX_PATH)
else:
    logging.info(f"FAISS index found at {INDEX_PATH}.")

def insert_face(name: str, vector: np.ndarray):
    logging.info(f"Inserting face for '{name}'.")
    index = faiss.read_index(INDEX_PATH)
    index.add(np.array([vector]))
    faiss.write_index(index, INDEX_PATH)
    logging.info(f"Face vector added to FAISS index.")

    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS faces (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT)")
        c.execute("INSERT INTO faces (name) VALUES (?)", (name,))
        conn.commit()
        logging.info(f"Metadata for '{name}' inserted into database.")

def search_face(query: np.ndarray):
    logging.info("Searching for face match.")
    index = faiss.read_index(INDEX_PATH)
    D, I = index.search(np.array([query]), k=1)

    if len(I[0]) == 0:
        logging.info("No match found in FAISS index.")
        return "No match", float("inf")

    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT name FROM faces WHERE id = ?", (I[0][0] + 1,))
        row = c.fetchone()
        if row:
            logging.info(f"Match found: {row[0]} with distance {D[0][0]:.4f}")
        else:
            logging.info("Match found in index, but no corresponding metadata in database.")
        return (row[0] if row else "Unknown", float(D[0][0]))