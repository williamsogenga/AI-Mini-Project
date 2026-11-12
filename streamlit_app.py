"""Local Streamlit interface for the lung-sound-classification project.

Place this file next to demo.py, then run:
    .\.venv\Scripts\python.exe -m streamlit run streamlit_app.py

Educational/research prototype only; never use its output for diagnosis.
"""
from collections import Counter
from pathlib import Path
import tempfile

import pandas as pd
import streamlit as st

from demo import classify_segments, load_models, segment_audio
from src.config import CYCLE_CLASS_NAMES

st.set_page_config(page_title="Respiratory Sound Classifier", page_icon="🫁", layout="centered")
st.title("Respiratory Sound Classification")
st.write("Upload a respiratory sound recording (.wav) to classify short audio segments using the trained machine-learning model.")
st.warning("Research and educational demonstration only. This system is not a medical diagnostic tool and its predictions must not guide care.")

@st.cache_resource
def get_models():
    return load_models()

uploaded = st.file_uploader("Choose a WAV audio file", type=["wav"])
if uploaded is not None:
    if uploaded.size > 50 * 1024 * 1024:
        st.error("File is too large. Please choose a WAV file smaller than 50 MB.")
        st.stop()
    audio_bytes = uploaded.getvalue()
    st.audio(audio_bytes, format="audio/wav")
    if st.button("Classify recording", type="primary"):
        temp_path = None
        try:
            with st.spinner("Processing audio and predicting sound classes..."):
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    temp_path = Path(tmp.name)
                    tmp.write(audio_bytes)
                model, scaler = get_models()
                # Without expert annotations, demo.py uses overlapping sliding windows.
                segments, _ = segment_audio(str(temp_path), annotation_path=None)
                if not segments:
                    st.error("No segments were generated. Try a recording at least 3 seconds long.")
                    st.stop()
                results = classify_segments(segments, model, scaler)
            counts = Counter(item["class_name"] for item in results)
            st.subheader("Prediction summary")
            st.metric("Audio segments classified", len(results))
            chart = pd.DataFrame({"Sound class": CYCLE_CLASS_NAMES,
                                  "Segments": [counts.get(name, 0) for name in CYCLE_CLASS_NAMES]})
            st.bar_chart(chart.set_index("Sound class"))
            st.dataframe(chart, hide_index=True, use_container_width=True)
            st.subheader("Segment-by-segment predictions")
            st.dataframe(pd.DataFrame([
                {"Start (s)": round(r["start"], 1), "End (s)": round(r["end"], 1),
                 "Predicted class": r["class_name"]} for r in results
            ]), hide_index=True, use_container_width=True)
            st.caption("Predictions are for overlapping windows of audio, not confirmed respiratory cycles. No ground-truth labels are available for uploaded files.")
        except Exception as exc:
            st.error(f"Classification failed: {exc}")
            st.info("Check that outputs/models/best_ml_model.joblib and scaler.joblib exist and your WAV file is valid.")
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)
