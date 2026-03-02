import os
from typing import List, Optional, Tuple

import joblib
import pandas as pd
import streamlit as st

from utils.config import (
    MODELS_DIR,
    DATA_DIR,
    MODEL_FILENAME,
    ENCODER_FILENAME,
    FEATURES_FILENAME,
    SAMPLE_DATA_FILENAME,
)


@st.cache_resource
def load_artifacts() -> Tuple[object, Optional[object], List[str]]:
    model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
    encoder_path = os.path.join(MODELS_DIR, ENCODER_FILENAME)
    features_path = os.path.join(DATA_DIR, FEATURES_FILENAME)

    missing = [p for p in [model_path, features_path] if not os.path.exists(p)]
    if missing:
        st.error(
            "Required ML artifacts are missing.\n\n"
            + "\n".join(f"- `{path}` not found" for path in missing)
        )
        st.stop()

    try:
        model = joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    encoder = None
    if os.path.exists(encoder_path):
        try:
            encoder = joblib.load(encoder_path)
        except Exception as e:
            st.warning(f"Encoder found but could not be loaded: {e}")

    features: List[str] = pd.read_csv(features_path, header=None)[0].tolist()

    return model, encoder, features


@st.cache_data
def load_sample_data() -> Optional[pd.DataFrame]:
    sample_path = os.path.join(DATA_DIR, SAMPLE_DATA_FILENAME)
    if not os.path.exists(sample_path):
        return None

    try:
        return pd.read_csv(sample_path)
    except Exception as e:
        st.warning(f"Could not read sample data: {e}")
        return None
