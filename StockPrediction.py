import os
import streamlit as st
import yt_dlp
import whisper
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from alpha_vantage.timeseries import TimeSeries

# --- Page Setup ---
st.set_page_config(page_title="Speech-Driven Stock Analysis", page_icon="📊", layout="wide")

# --- Theme State ---
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "transcript" not in st.session_state:
    st.session_state.transcript = None
    st.session_state.sentiment = None
    st.session_state.key_sentences = None

# Clear previous results if input type changes
if "last_input_type" not in st.session_state:
    st.session_state.last_input_type = None

# --- Theme Configuration ---
dark_mode = st.session_state.dark_mode
primary_color = "#ffffff" if dark_mode else "#202123"
secondary_color = "#d0d0d0" if dark_mode else "#6b6b6b"
background_color = "#0e1117" if dark_mode else "#ffffff"
input_color = "#1c1f26" if dark_mode else "#f5f5f5"
text_color = "#f0f0f0" if dark_mode else "#000000"
radio_color = "#ffffff" if dark_mode else "#333333"

# --- Custom Styles ---
st.markdown(f"""
    <style>
    .stApp {{ background-color: {background_color}; color: {text_color}; }}
    .big-title {{ font-size: 32px; font-weight: bold; color: {primary_color}; text-align: center; margin-top: 10px; margin-bottom: 5px; }}
    .sub-title {{ font-size: 18px; color: {secondary_color}; text-align: center; margin-bottom: 5px; }}
    .disclaimer {{ font-size: 14px; color: #FF5757; text-align: center; margin-top: 20px; }}
    .stTextInput input, .stFileUploader, .stButton button {{ background-color: {input_color}; color: {text_color}; border-radius: 5px; }}
    .stRadio > div {{ flex-direction: row !important; gap: 1.5rem; justify-content: center; }}
    .stRadio label {{ color: {radio_color}; font-weight: 600; font-size: 16px; }}
    .input-button-container {{ display: flex; justify-content: center; align-items: center; gap: 0.5rem; margin-bottom: 0; margin-top: -10px; }}
    summary {{ font-size: 22px !important; font-weight: bold; }}
    </style>
""", unsafe_allow_html=True)

# --- Theme Toggle ---
col1, col2 = st.columns([10, 1])
with col2:
    icon = "🌞" if dark_mode else "🌙"
    if st.button(icon, key="theme_switch", help="Toggle theme"):
        st.session_state.dark_mode = not st.session_state.dark_mode
        st.rerun()

# --- Load Resources ---
@st.cache_resource
def load_sentiment_model():
    return load_model("speech_sentiment_model.keras")

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

model = load_sentiment_model()
tokenizer = load_tokenizer()

# --- Functions ---
def transcribe_audio(audio_path):
    try:
        whisper_model = whisper.load_model("tiny.en")
        result = whisper_model.transcribe(audio_path)
        return result["text"]
    except Exception as e:
        st.error(f"❌ Whisper error: {e}")
        return ""

def predict_sentiment(text):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=200)
    prediction = model.predict(padded)
    return ["Bearish 📉", "Neutral 📊", "Bullish 📈"][np.argmax(prediction)]

def extract_key_sentences(transcript):
    sentences = transcript.split(". ")
    return sentences[:2] if len(sentences) > 1 else sentences

def download_youtube_audio(youtube_url, output_path="downloaded_audio.mp3"):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path.replace(".mp3", ""),
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3', 'preferredquality': '192'}],
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])
        if os.path.exists(output_path + ".mp3") and not os.path.exists(output_path):
            os.rename(output_path + ".mp3", output_path)
        return output_path
    except Exception as e:
        st.error(f"❌ YouTube download error: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_sp500_chart():
    ts = TimeSeries(key="IM3YU1NKAZ8HT60", output_format='pandas')
    data, _ = ts.get_monthly(symbol="SPY")
    data = data.sort_index()
    data = data[data.index >= '2025-01-01']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['4. close'], mode='lines', name='S&P 500'))
    return fig

# --- UI ---
st.markdown("""
<h1 class="big-title">📊 Speech-Driven Predictive Analysis of the US Stock Market</h1>
<p class="sub-title">🔍 Analyze speeches from financial leaders and predict stock market impact.</p>
""", unsafe_allow_html=True)

input_type = st.radio("Select Input Type:", ["YouTube Link", "Upload MP3", "Upload Video"], horizontal=True)

if st.session_state.last_input_type != input_type:
    st.session_state.last_input_type = input_type
    st.session_state.transcript = None
    st.session_state.sentiment = None
    st.session_state.key_sentences = None

st.markdown("<div class='input-button-container'>", unsafe_allow_html=True)
if input_type == "YouTube Link":
    user_input = st.text_input("", placeholder="Paste YouTube link here…", label_visibility="collapsed")
else:
    uploaded_file = st.file_uploader("", type=["mp3", "mp4", "mov", "avi"], label_visibility="collapsed")
st.markdown("</div>", unsafe_allow_html=True)

analyze_button = st.button("Analyze Speech")

if analyze_button:
    file_path = None
    if input_type == "YouTube Link" and user_input:
        file_path = download_youtube_audio(user_input)
    elif uploaded_file:
        file_path = uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    if file_path:
        st.info("⏳ Transcribing and analyzing…")
        transcript = transcribe_audio(file_path)
        st.session_state.transcript = transcript
        st.session_state.key_sentences = extract_key_sentences(transcript)
        st.session_state.sentiment = predict_sentiment(transcript)
        st.success("✅ Analysis complete!")

# --- Results ---
if st.session_state.transcript:
    with st.expander("📝 Key Transcript Insights"):
        for sentence in st.session_state.key_sentences:
            st.write(f"- {sentence}")

    with st.expander(f"📊 Sentiment Prediction: {st.session_state.sentiment}"):
        sentiment_map = {"Bearish 📉": 0, "Neutral 📊": 50, "Bullish 📈": 100}
        fig_meter = go.Figure(go.Indicator(
            mode="gauge+number",
            value=sentiment_map[st.session_state.sentiment],
            title={"text": "Market Sentiment"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "lightblue"}},
        ))
        st.plotly_chart(fig_meter)

    with st.expander("📈 Real-Time S&P 500 Market Trend"):
        fig = fetch_sp500_chart()
        if fig:
            st.plotly_chart(fig)

    st.markdown(
        '<p class="disclaimer">⚠️ This is not financial advice. Please do your own research before making investment decisions.</p>',
        unsafe_allow_html=True
    )
