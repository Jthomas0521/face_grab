from deepface import DeepFace
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

def get_embedding(image_path: str, model_name="Facenet"):
    logging.info(f"Extracting embedding for image: {image_path} using model: {model_name}")
    result = DeepFace.represent(img_path=image_path, model_name=model_name, enforce_detection=True)[0]
    embedding = np.array(result["embedding"], dtype=np.float32)
    logging.info(f"Embedding extracted for image: {image_path}")
    return embedding