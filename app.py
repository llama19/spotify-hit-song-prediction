import streamlit as st
import joblib
import numpy as np

model = joblib.load("hit_song_model.pkl")

st.title("Spotify Hit Song Predictor")

st.write("Enter Spotify audio features to predict whether a song will become a hit.")

danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
energy = st.slider("Energy", 0.0, 1.0, 0.5)
loudness = st.slider("Loudness", -60.0, 0.0, -10.0)
speechiness = st.slider("Speechiness", 0.0, 1.0, 0.1)
acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)
instrumentalness = st.slider("Instrumentalness", 0.0, 1.0, 0.0)
liveness = st.slider("Liveness", 0.0, 1.0, 0.1)
valence = st.slider("Valence", 0.0, 1.0, 0.5)
tempo = st.slider("Tempo", 50.0, 200.0, 120.0)
if st.button("Predict Hit Song"):

    features = np.array([[danceability, energy, loudness, speechiness,
                          acousticness, instrumentalness, liveness,
                          valence, tempo]])

    prediction = model.predict(features)
    prob = model.predict_proba(features)[0][1]

    st.subheader("Prediction Result")

    st.write(f"Hit Probability: {prob*100:.2f}%")

    if prob > 0.5:
        st.success("🎵 Likely to be a HIT")
    else:
        st.warning("❌ Unlikely to be a hit")