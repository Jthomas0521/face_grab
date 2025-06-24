from deepface import DeepFace
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def get_embedding(image_path: str):
    try:
        logging.info(f"Extracting embedding for image: {image_path} using Deepface models")

        results = DeepFace.represent(img_path=image_path, enforce_detection=True)
        
        embeddings = [np.array(res["embedding"], dtype=np.float32) for res in results]
        raw_embedding = np.mean(embeddings, axis=0) 
        
        normalized_embedding = raw_embedding / np.linalg.norm(raw_embedding)
        logging.info(f"Embedding extracted for image: {image_path}")
        return normalized_embedding
    
    except ValueError as e:
        logging.warning(f"Face not detected: {e}")
        return None