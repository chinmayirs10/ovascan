"""Streamlit demo: upload an ovarian ultrasound image, get a result band and a lifestyle plan.

Run:  streamlit run app.py
"""

from pathlib import Path

import streamlit as st
from PIL import Image

from pcos_screen.lifestyle import Profile, build_plan
from pcos_screen.model import HIGH, LOW, band_for

WEIGHTS = Path(__file__).parent / "models" / "pcos_mobilenetv3.pt"

st.set_page_config(page_title="PCOS Ultrasound Screening (research prototype)", page_icon="🩺")

st.title("PCOS ultrasound screening")
st.caption("Research prototype · work in progress")

st.warning(
    "**Not a medical device and not a diagnosis.** This is a student learning project trained "
    "on a small public dataset with known quality problems (see the README). PCOS can only be "
    "diagnosed by a clinician using symptoms, blood tests and imaging together. "
    "Do not upload images containing personal data.",
    icon="⚠️",
)


@st.cache_resource
def load_predictor():
    from pcos_screen.model import Predictor
    return Predictor(WEIGHTS) if WEIGHTS.exists() else None


predictor = load_predictor()
if predictor is None:
    st.error("No trained model found. Run `python scripts/audit_dataset.py` and `python scripts/train.py` first.")
    st.stop()

uploaded = st.file_uploader("Ovarian ultrasound image (JPG/PNG)", type=["jpg", "jpeg", "png"])

with st.expander("Optional: tell us a bit about your habits for a more personal plan"):
    c1, c2 = st.columns(2)
    activity = c1.number_input("Active minutes per week", 0, 2000, value=None, step=10)
    strength = c1.number_input("Strength-training days per week", 0, 7, value=None)
    sleep = c1.number_input("Average sleep (hours)", 0.0, 14.0, value=None, step=0.5)
    irregular = c2.checkbox("My periods are irregular")
    androgen = c2.checkbox("I have acne or excess hair growth")
    mood = c2.checkbox("I often feel low or anxious")

if uploaded:
    image = Image.open(uploaded)
    col_img, col_res = st.columns([1, 1])
    col_img.image(image, caption="Uploaded image", width="stretch")

    prob = predictor.predict_proba(image)
    band = band_for(prob)
    colour = {"normal": "green", "inconclusive": "orange", "pcos_like": "red"}[band.key]
    col_res.markdown(f"### :{colour}[{band.title}]")
    col_res.write(band.explanation)
    col_res.progress(prob, text=f"Model score for PCOS-like pattern: {prob:.0%}")
    col_res.caption(f"Below {LOW:.0%}: no PCOS-like features · {LOW:.0%}-{HIGH:.0%}: inconclusive · "
                    f"above {HIGH:.0%}: PCOS-like features")

    profile = Profile(
        activity_min_per_week=activity, strength_days_per_week=strength, sleep_hours=sleep,
        irregular_cycles=irregular, acne_or_excess_hair=androgen, low_mood_or_anxiety=mood,
    )
    plan = build_plan(band.key, profile)

    st.divider()
    st.subheader(plan.headline)
    st.info(plan.see_doctor, icon="🩺")
    for section in plan.sections:
        st.markdown(f"**{section.title}**")
        st.markdown("\n".join(f"- {item}" for item in section.items))
    st.caption(f"Source: {plan.reference}")
