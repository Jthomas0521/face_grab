import faiss
import numpy as np
import sqlite3
import os
import logging

logging.basicConfig(level=logging.INFO)

DB_PATH = "face_metadata.db"
INDEX_PATH = "face_index.faiss"
DIM_PATH = "faiss_dim.txt"
THRESHOLD = 3.5


def _init_index(vector: np.ndarray):
    dim = vector.shape[0]
    logging.info(f"Initializing FAISS index with dim: {dim}")
    index = faiss.IndexFlatL2(dim)
    faiss.write_index(index, INDEX_PATH)

    # Save dimension to file
    with open(DIM_PATH, "w") as f:
        f.write(str(dim))
    return index


def _get_index(vector: np.ndarray):
    if not os.path.exists(INDEX_PATH):
        return _init_index(vector)

    index = faiss.read_index(INDEX_PATH)

    expected_dim = vector.shape[0]
    if index.d != expected_dim:
        raise ValueError(f"Vector dimension {expected_dim} does not match FAISS index dimension {index.d}")

    return index


def insert_face(name: str, vector: np.ndarray):
    logging.info(f"Inserting face for '{name}'.")

    vector = np.array(vector, dtype=np.float32)
    index = _get_index(vector)

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

    if not os.path.exists(INDEX_PATH):
        logging.warning("FAISS index does not exist.")
        return "No match", None

    index = faiss.read_index(INDEX_PATH)
    query = np.array(query, dtype=np.float32)

    if query.shape[0] != index.d:
        raise ValueError(f"Query dimension {query.shape[0]} does not match FAISS index dimension {index.d}")

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
