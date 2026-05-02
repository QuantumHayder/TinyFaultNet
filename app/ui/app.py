"""
Streamlit frontend for TinyFaultNet.
Supports multiple models — user picks from a dropdown.

Run:
    cd app/ui
    streamlit run app.py --server.port 8501
"""

import streamlit as st
import requests
import os

# ─── Config ───────────────────────────────────────────────────────────────────

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="TinyFaultNet - Machine Audio Classifier",
    page_icon="🔊",
    layout="centered",
)

st.markdown("""
<style>
    .stMetric { border: 1px solid #e0e0e0; border-radius: 8px; padding: 10px; }
</style>
""", unsafe_allow_html=True)


# ─── Header ───────────────────────────────────────────────────────────────────

st.title("🔊 TinyFaultNet")
st.markdown("Machine audio fault detection — upload a recording to classify it.")
st.divider()


# ─── API health check ────────────────────────────────────────────────────────

@st.cache_data(ttl=30)
def check_api():
    try:
        health = requests.get(f"{API_URL}/health", timeout=5).json()
        models = requests.get(f"{API_URL}/models", timeout=5).json()
        return health, models
    except requests.exceptions.ConnectionError:
        return None, None
    except Exception:
        return None, None


health, models_info = check_api()

if health is None:
    st.error(
        f"Cannot connect to API at `{API_URL}`.\n\n"
        "Make sure the FastAPI server is running:\n"
        "```\ncd app/api\nuvicorn main:app --port 8000\n```"
    )
    st.stop()

if health.get("models_loaded", 0) == 0:
    st.warning("No models loaded. Place .joblib bundles in the app/ directory.")
    st.stop()

st.success(f"✅ API connected — {health['models_loaded']} model(s) loaded.")


# ─── Model selection ──────────────────────────────────────────────────────────

available_models = models_info["models"]
model_names = [m["name"] for m in available_models]

# Sidebar for model selection and info
with st.sidebar:
    st.header("⚙️ Model Selection")

    selected_model = st.selectbox(
        "Choose a model",
        model_names,
        index=model_names.index(models_info["active"]) if models_info["active"] in model_names else 0,
    )

    # Show selected model info
    selected_info = next(m for m in available_models if m["name"] == selected_model)

    st.markdown("### Model Details")
    st.markdown(f"**Features:** {selected_info['expected_features']}")
    st.markdown(f"**Classes:** {', '.join(selected_info['classes'])}")

    cfg = selected_info["config"]
    st.markdown(f"**Sample rate:** {cfg['sr']} Hz")
    st.markdown(f"**MFCC coefficients:** {cfg['n_mfcc']}")
    st.markdown(f"**Mel filters:** {cfg['n_filters']}")
    st.markdown(f"**FFT size:** {cfg['NFFT']}")

    # Pipeline indicator
    if selected_info["expected_features"] == 195:
        st.info("📋 Legacy pipeline (MFCC only)")
    else:
        st.info("🚀 Augmented pipeline (MFCC + Mel)")


# ─── File upload ──────────────────────────────────────────────────────────────

uploaded_file = st.file_uploader(
    "Choose an audio file",
    type=["wav", "mp3", "flac", "ogg", "m4a"],
    help="Supported: WAV, MP3, FLAC, OGG, M4A. Max 50 MB.",
)

if uploaded_file is not None:
    # Audio player
    st.audio(uploaded_file)

    size_mb = uploaded_file.size / (1024 * 1024)
    st.caption(f"📁 **{uploaded_file.name}** — {size_mb:.2f} MB")

    if size_mb > 50:
        st.error("File too large. Please upload a file under 50 MB.")
        st.stop()

    # Classify button
    if st.button("🔍 Classify", type="primary", use_container_width=True):
        with st.spinner(f"Analyzing with **{selected_model}**..."):
            try:
                response = requests.post(
                    f"{API_URL}/predict",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue())},
                    params={"model": selected_model},
                    timeout=120,
                )

                if response.status_code == 200:
                    result = response.json()
                    st.divider()

                    # ── Main result ───────────────────────────────────────
                    predicted = result["predicted_class"]

                    if predicted.lower() in ("normal", "healthy", "ok", "m1_normal", "m2_normal", "m3_normal"):
                        st.success(f"### ✅ {predicted.upper()}")
                    else:
                        st.error(f"### ⚠️ {predicted.upper()}")

                    # ── Metrics ───────────────────────────────────────────
                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(label="Predicted Class", value=predicted.upper())

                    with col2:
                        if "decision_score" in result:
                            st.metric(
                                label="Decision Score",
                                value=f"{result['decision_score']:.4f}",
                            )
                        elif "decision_scores" in result:
                            scores = result["decision_scores"]
                            top_class = max(scores, key=scores.get)
                            st.metric(
                                label=f"Top Score ({top_class})",
                                value=f"{scores[top_class]:.4f}",
                            )

                    with col3:
                        st.metric(
                            label="Inference Time",
                            value=f"{result['inference_time_sec']:.3f}s",
                        )

                    # ── Model used badge ──────────────────────────────────
                    st.caption(f"Model: **{result['model_used']}**")

                    # ── Multiclass scores breakdown ───────────────────────
                    if "decision_scores" in result:
                        st.divider()
                        st.subheader("Decision Scores")
                        scores = result["decision_scores"]
                        sorted_scores = dict(
                            sorted(scores.items(), key=lambda x: x[1], reverse=True)
                        )
                        for cls_name, score in sorted_scores.items():
                            col_name, col_bar = st.columns([1, 3])
                            with col_name:
                                st.write(f"**{cls_name}**")
                            with col_bar:
                                all_vals = list(scores.values())
                                min_s, max_s = min(all_vals), max(all_vals)
                                rng = max_s - min_s if max_s != min_s else 1.0
                                normalized = (score - min_s) / rng
                                st.progress(normalized)

                    # ── Raw response ──────────────────────────────────────
                    with st.expander("📋 Raw API Response"):
                        st.json(result)

                else:
                    error_detail = response.json().get("detail", "Unknown error")
                    st.error(f"API error ({response.status_code}): {error_detail}")

            except requests.exceptions.Timeout:
                st.error("Request timed out.")
            except requests.exceptions.ConnectionError:
                st.error("Lost connection to API.")
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")


# ─── Footer ───────────────────────────────────────────────────────────────────

st.divider()
st.caption("TinyFaultNet — Pattern Recognition Project (CMPS450)")