import logging
from deepface import DeepFace

logging.basicConfig(level=logging.INFO)

def download_all_models():
    models = ["VGG-Face", "Facenet", "OpenFace", "DeepFace", "ArcFace", "Dlib"]
    for model in models:
        logging.info(f"Downloading and caching model: {model}")
        DeepFace.build_model(model)
    logging.info("All models downloaded and cached.")

if __name__ == "__main__":
    download_all_models()