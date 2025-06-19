import faiss
import numpy as np
import sqlite3
import os
import logging

logging.basicConfig(level=logging.INFO)

DB_PATH = "face_metadata.db"
INDEX_PATH = "face_index.faiss"
DIM = 128
THRESHOLD = 1.3

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
        c.execute("CREATE TABLE IF NOT EXISTS face_mappings (faiss_idx INTEGER, face_id INTEGER)")

        c.execute("INSERT INTO faces (name) VALUES (?)", (name,))
        face_id = c.lastrowid

        faiss_idx = index.ntotal - 1
        c.execute("INSERT INTO face_mappings (faiss_idx, face_id) VALUES (?, ?)", (faiss_idx, face_id))

        conn.commit()
        logging.info(f"Metadata for '{name}' inserted into database.")


def search_face(query: np.ndarray):
    logging.info("Searching for face match.")

    index = faiss.read_index(INDEX_PATH)
    D, I = index.search(np.array([query]), k=1)
    distance = D[0][0]
    faiss_idx = int(I[0][0])

    logging.info(f"Distance: {distance}")
    logging.info(f"Using threshold: {THRESHOLD}")

    if len(I[0]) == 0 or distance > THRESHOLD:
        logging.info("No match found in FAISS index.")
        return "No match", None

    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT face_id FROM face_mappings WHERE faiss_idx = ?", (faiss_idx,))
        result = c.fetchone()

        if not result:
            logging.info("Match found in FAISS, but no mapping in DB.")
            return "Unknown", distance

        face_id = result[0]
        c.execute("SELECT name FROM faces WHERE id = ?", (face_id,))
        row = c.fetchone()

        if row:
            logging.info(f"Match found: {row[0]} with distance {distance:.4f}")
            return row[0], distance
        else:
            logging.info("Mapping found, but name not found.")
            return "Unknown", distance
