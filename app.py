import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

# Load the model and tokenizer ONCE
model_path = "Karan2805-glitch/brand-sentiment-bert"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Get device (optional, only for inputs)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Sentiment prediction function
def predict_sentiment(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        confidence, prediction = torch.max(probs, dim=-1)
        confidence = confidence.item()
        prediction = prediction.item()

    if confidence < 0.6:
        return "Uncertain", confidence
    else:
        return ("Positive" if prediction == 1 else "Negative"), confidence

# Streamlit App
st.title("Brand Sentiment Analyzer")
user_input = st.text_input("Enter text for sentiment prediction:")
if user_input:
    prediction, confidence = predict_sentiment(user_input)
    st.success(f"Prediction: {prediction} (Confidence: {confidence:.2f})")

st.subheader("Analyze a File (CSV Upload)")
uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Sample Data:", df.head())

    if 'text' not in df.columns:
        st.error("CSV must have a 'text' column!")
    else:
        st.write("Running sentiment analysis on uploaded data...")

        sentiments = []
        for text in df['text']:
            if pd.isnull(text):
                sentiments.append("Unknown")
            else:
                sentiment, confidence = predict_sentiment(str(text))
                sentiments.append(sentiment)

        df['Sentiment'] = sentiments
        st.write(df)

        # Bar chart
        sentiment_counts = df['Sentiment'].value_counts()
        st.bar_chart(sentiment_counts)
