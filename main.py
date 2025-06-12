from fastapi import FastAPI
from src.routes import router

app = FastAPI(title="Face Grab API")
app.include_router(router)
