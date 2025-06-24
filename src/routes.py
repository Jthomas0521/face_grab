from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from src.logic import get_embedding
from src.db import insert_face, search_face
import os
import shutil
import uuid
import logging

logging.basicConfig(level=logging.INFO)

router = APIRouter()
UPLOAD_DIR = "images"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/register/")
async def register_face(name: str = Form(...), file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in [".jpg", ".jpeg", ".png", ".webp"]:
        logging.warning(f"Unsupported file type: {ext} for file '{file.filename}'")
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a .jpg, .jpeg, .png, or .webp image.")
    
    image_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}{ext}")
    logging.info(f"Registering face for '{name}'. Saving uploaded image to {image_path}")
    with open(image_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    embedding = get_embedding(image_path)
    if embedding is None:
        logging.warning(f"No face detected in image for '{name}'. Registration aborted.")
        raise HTTPException(status_code=400, detail="No face detected in the uploaded image.")

    insert_face(name, embedding)
    logging.info(f"Face for '{name}' registered successfully.")
    return {"status": "success", "name": name}

@router.post("/recognize/")
async def recognize_face(file: UploadFile = File(...)):
    image_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}.jpg")
    logging.info(f"Recognizing face. Saving uploaded image to {image_path}")
    with open(image_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    embedding = get_embedding(image_path)
    if embedding is None:
        logging.warning("No face detected during recognition. Aborting.")
        raise HTTPException(status_code=400, detail="No face detected in the uploaded image.")

    name, distance = search_face(embedding)
    logging.info(f"Recognition result: match={name}, distance={distance}")
    return {"match": name, "distance": float(distance) if distance is not None else None}
