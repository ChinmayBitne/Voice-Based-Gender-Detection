import os
import wave
import librosa
import pyaudio
import numpy as np
import streamlit as st
from utils import create_model

THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000
MAX_RECORDING_TIME = 10  # Maximum recording time in seconds
MIN_RECORDING_TIME = 4  # Minimum recording time in seconds

def is_silent(snd_data):
    return max(snd_data) < THRESHOLD

def record_to_file(path, max_duration):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
                    input=True, frames_per_buffer=CHUNK_SIZE)

    frames = []
    recorded_frames = 0

    while recorded_frames < max_duration * RATE / CHUNK_SIZE:
        data = stream.read(CHUNK_SIZE)
        frames.append(data)
        recorded_frames += 1

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

def record_audio():
    with st.spinner("Recording..."):
        path = "temp_audio.wav"
        record_to_file(path, MAX_RECORDING_TIME)
    st.success("Recording complete.")
    return path

def extract_feature(file_name, **kwargs):
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    
    X, sample_rate = librosa.core.load(file_name)
    if chroma or contrast:
        stft = np.abs(librosa.stft(X))
    features = []
    if mfcc:
        mfccs = librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40)
        features.append(np.mean(mfccs, axis=1))
    if chroma:
        chroma = librosa.feature.chroma_stft(S=stft, sr=sample_rate)
        features.append(np.mean(chroma, axis=1))
    if mel:
        mel = librosa.feature.melspectrogram(y=X, sr=sample_rate)
        features.append(np.mean(mel, axis=1))
    if contrast:
        contrast = librosa.feature.spectral_contrast(S=stft, sr=sample_rate)
        features.append(np.mean(contrast, axis=1))
    if tonnetz:
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate)
        features.append(np.mean(tonnetz, axis=1))
    
    return np.concatenate(features)

def main():
    st.title("Voice Based Gender Recognition")
    st.write("This app recognizes gender from audio.")

    # Dropdown menu for recording and uploading file options
    option = st.selectbox("Select Option", ["Record Audio", "Upload Audio File"])

    # Handle selected option
    if option == "Record Audio":
        if st.button("Record", key="record"):
            file = record_audio()
    else:
        uploaded_file = st.file_uploader("Upload Audio File (WAV format)", accept_multiple_files=False)
        if uploaded_file is not None:
            with open("temp_audio.wav", "wb") as f:
                f.write(uploaded_file.getvalue())
            file = "temp_audio.wav"

    # Process the file if exists
    if "file" in locals():
        model = create_model()
        model.load_weights("results/model.h5")

        if os.path.exists(file):
            features = extract_feature(file, mel=True).reshape(1, -1)
            male_prob = model.predict(features)[0][0]
            female_prob = 1 - male_prob
            gender = "Male" if male_prob > female_prob else "Female"

            st.subheader("Result")
            st.success(gender)
            st.subheader("Probabilities")
            col1, col2 = st.columns(2)
            with col1:
                st.text(f'Male {male_prob * 100 :.2f}%')
                st.progress(int(male_prob * 100))
            with col2:
                st.text(f'Female {female_prob * 100 :.2f}%')
                st.progress(int(female_prob * 100))
        os.remove('temp_audio.wav')

if __name__ == "__main__":
    main()