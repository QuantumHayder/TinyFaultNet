"""
FastAPI backend for Machine Audio Classification.
Supports multiple models — user selects which one to use.

Run:
    cd app/api
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import shutil
import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from model import AudioClassifier


# ─── Lifespan ─────────────────────────────────────────────────────────────────

classifier = AudioClassifier()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading models...")
    classifier.load_all()
    yield


app = FastAPI(
    title="TinyFaultNet — Machine Audio Classifier",
    description="SVM-based machine sound classification API with multiple model support",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Endpoints ────────────────────────────────────────────────────────────────

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
MAX_FILE_SIZE_MB = 50


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "models_loaded": len(classifier.models),
        "active_model": classifier.active_model,
    }


@app.get("/models")
def list_models():
    """List all available models and their info."""
    return {
        "active": classifier.active_model,
        "models": classifier.list_models(),
    }


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model: str = Query(default=None, description="Model name to use. Defaults to active model."),
):
    """
    Upload an audio file and get a classification prediction.
    Optionally specify which model to use via the 'model' query parameter.
    """
    # Validate extension
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    # Validate model name if provided
    if model and model not in classifier.models:
        available = list(classifier.models.keys())
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model}' not found. Available: {available}",
        )

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        size_mb = os.path.getsize(tmp_path) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"File too large ({size_mb:.1f} MB). Max: {MAX_FILE_SIZE_MB} MB",
            )

        result = classifier.predict(tmp_path, model_name=model)
        result["filename"] = file.filename
        result["file_size_mb"] = round(size_mb, 2)
        return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)