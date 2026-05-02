"""
Model loading and inference.
Supports multiple model bundles — user selects which model to use.
Auto-detects the correct preprocessing pipeline from model.n_features_in_.
"""

import os
import time
import numpy as np
import joblib
from pathlib import Path
from typing import Any

from preprocessing import preprocess_audio


# ─── Configuration ────────────────────────────────────────────────────────────

# Directory containing .joblib bundles (app/ folder)
MODELS_DIR = Path(__file__).parent.parent

# HuggingFace repo (fallback)
HF_REPO_ID = os.getenv(
    "HF_REPO_ID",
    "your-username/machine-audio-classifier",
)


# ─── Single model container ──────────────────────────────────────────────────

class LoadedModel:
    """Holds a loaded model bundle and its metadata."""

    def __init__(self, name: str, bundle: dict):
        self.name         = name
        self.model        = bundle["model"]
        self.scaler       = bundle["scaler"]
        self.class_names  = bundle["class_names"]
        self.label_map    = bundle["label_map"]
        self.cfg          = bundle["extractor_cfg"]
        self.n_features   = self.model.n_features_in_

    def info(self) -> dict:
        return {
            "name": self.name,
            "expected_features": self.n_features,
            "classes": self.class_names,
            "config": self.cfg,
        }


# ─── Multi-model manager ─────────────────────────────────────────────────────

class AudioClassifier:
    def __init__(self):
        self.models: dict[str, LoadedModel] = {}
        self.active_model: str = ""

    def load_all(self):
        """
        Scan the app/ directory for all .joblib bundle files and load them.
        The first one found becomes the default active model.
        """
        bundle_files = sorted(MODELS_DIR.glob("*.joblib"))

        if not bundle_files:
            print(f"No .joblib files found in {MODELS_DIR}")
            print("Attempting HuggingFace download...")
            self._download_from_hf()
            bundle_files = sorted(MODELS_DIR.glob("*.joblib"))

        for path in bundle_files:
            try:
                bundle = joblib.load(path)
                # Validate bundle has required keys
                required = {"model", "scaler", "class_names", "label_map", "extractor_cfg"}
                if not required.issubset(bundle.keys()):
                    print(f"  Skipping {path.name}: missing keys {required - bundle.keys()}")
                    continue

                name = path.stem  # filename without extension
                loaded = LoadedModel(name, bundle)
                self.models[name] = loaded
                print(f"  Loaded: {name} ({loaded.n_features} features, "
                      f"{len(loaded.class_names)} classes)")
            except Exception as e:
                print(f"  Failed to load {path.name}: {e}")

        if self.models:
            self.active_model = list(self.models.keys())[0]
            print(f"\nActive model: {self.active_model}")
        else:
            raise RuntimeError("No valid model bundles found.")

    def _download_from_hf(self):
        """Download bundle from HuggingFace as fallback."""
        try:
            from huggingface_hub import hf_hub_download
            path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename="machine_listener_bundle.joblib",
                local_dir=str(MODELS_DIR),
            )
            print(f"  Downloaded to {path}")
        except ImportError:
            raise RuntimeError(
                "No local bundles found and huggingface_hub not installed."
            )

    def list_models(self) -> list[dict]:
        """Return info about all loaded models."""
        return [m.info() for m in self.models.values()]

    def set_active(self, name: str):
        """Switch the active model."""
        if name not in self.models:
            available = list(self.models.keys())
            raise ValueError(f"Model '{name}' not found. Available: {available}")
        self.active_model = name

    def predict(self, audio_path: str, model_name: str | None = None) -> dict:
        """
        Run inference with the specified (or active) model.
        Auto-selects the correct preprocessing pipeline.
        """
        name = model_name or self.active_model
        if name not in self.models:
            raise ValueError(f"Model '{name}' not loaded.")

        m = self.models[name]
        t_start = time.perf_counter()

        # Preprocess — pipeline auto-selected by expected feature count
        features = preprocess_audio(
            audio_path,
            expected_features=m.n_features,
            sr=m.cfg["sr"],
            n_mfcc=m.cfg["n_mfcc"],
            n_mels=m.cfg["n_filters"],
            n_fft=m.cfg["NFFT"],
        )
        features = features.reshape(1, -1)

        # Scale + predict
        features_scaled = m.scaler.transform(features)
        prediction = m.model.predict(features_scaled)[0]
        class_name = m.class_names[int(prediction)]

        elapsed = time.perf_counter() - t_start

        # Decision scores
        decision = m.model.decision_function(features_scaled)[0]

        result = {
            "model_used": name,
            "predicted_label": int(prediction),
            "predicted_class": class_name,
            "inference_time_sec": round(elapsed, 3),
        }

        if np.isscalar(decision):
            result["decision_score"] = round(float(np.float64(decision)), 4)
        else:
            result["decision_scores"] = {
                m.class_names[i]: round(float(np.float64(d)), 4)
                for i, d in enumerate(decision)
            }

        return result