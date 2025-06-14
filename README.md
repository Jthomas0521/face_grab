# Face Grab – Facial Recognition API

Face Grab is a full-stack facial recognition API built with:

- **DeepFace** for face embedding extraction
- **FAISS** for fast vector similarity search
- **FastAPI** for lightweight REST endpoints
- **Docker** support for both CPU and GPU environments
- **SQLite** for metadata storage

---

## Features

- Register new faces with a name and photo
- Recognize unknown faces by matching against stored embeddings
- Store face vectors in a FAISS index and names in a SQLite database
- Easily deployable with Docker on macOS (CPU) or Linux (GPU)

---

## Project Structure

```
face_grab/
├── src/
│   ├── db.py                 # FAISS + SQLite logic
│   ├── logic.py              # DeepFace embedding extraction
│   └── routes.py             # FastAPI endpoints
├── main.py                   # App entry point
├── models/            
│   ├── download_models.py    # Pre-download DeepFace models
├── images/                   # Uploaded face images
├── requirements.txt          # Python dependencies
├── Dockerfile.cpu            # Dockerfile for macOS/CPU
├── Dockerfile.gpu            # Dockerfile for Linux/GPU
├── docker-compose.cpu.yml    # Compose config for CPU
├── docker-compose.gpu.yml    # Compose config for GPU
```

---

## API Endpoints

| Endpoint         | Method | Description                     |
|------------------|--------|---------------------------------|
| `/register/`     | POST   | Register a face with name + image |
| `/recognize/`    | POST   | Match unknown face against DB   |

---

## Example Usage (cURL)

### Register a Face
```bash
curl -X POST http://localhost:8000/register/ \
  -F "name=John Doe" \
  -F "file=@john.jpg"
```

### Recognize a Face
```bash
curl -X POST http://localhost:8000/recognize/ \
  -F "file=@mystery.jpg"
```

---

## Run with Docker

### On macOS / CPU:
```bash
docker compose -f docker-compose.cpu.yml up --build
```

Make sure you're using `Dockerfile.cpu`.

### On Linux / NVIDIA GPU:
```bash
docker compose -f docker-compose.gpu.yml up --build
```

Ensure you have [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) installed.

---

## Install Dependencies (For Local Dev)

```bash
pip install -r requirements.txt
```

Optional: preload DeepFace models
```bash
python download_models.py
```

---