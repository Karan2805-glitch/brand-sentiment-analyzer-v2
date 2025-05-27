import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd
import io

# Load the model and tokenizer
model_path = "Karan2805-glitch/brand-sentiment-bert"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Function to predict sentiment
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

# Streamlit App UI
st.title("Brand Sentiment Analyzer 🚀")
st.markdown("""
This app uses a **fine-tuned BERT model** to analyze the sentiment of text data (Positive, Negative, or Uncertain).
You can either:
- Enter a single text input, or
- Upload a CSV file containing multiple texts for batch analysis.
""")

st.markdown("Enter a text below or upload a CSV file to get sentiment predictions.")

# User input section (single text)
user_input = st.text_area("Enter text here:")

if st.button("Predict Sentiment"):
    if user_input.strip() != "":
        prediction, confidence = predict_sentiment(user_input)
        st.success(f"Prediction: {prediction} (Confidence: {confidence:.2f})")
    else:
        st.warning("Please enter some text!")

# CSV Upload Section
st.subheader("Analyze a File (CSV Upload)")
uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Sample Data:", df.head())

    if 'text' not in df.columns:
        st.error("CSV must have a 'text' column!")
    else:
        st.write("Running sentiment analysis on uploaded data...")

        # Run predictions on each row
        sentiments = []
        confidences = []
        for text in df['text']:
            if pd.isnull(text):
                sentiments.append("Unknown")
                confidences.append(0.0)
            else:
                sentiment, confidence = predict_sentiment(str(text))
                sentiments.append(sentiment)
                confidences.append(confidence)

        df['Sentiment'] = sentiments
        df['Confidence'] = [f"{c:.2f}" for c in confidences]
        st.write(df)
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download Predictions as CSV",
            data=csv,
            file_name='sentiment_predictions.csv',
            mime='text/csv',
            )

        # Show bar chart
        sentiment_counts = df['Sentiment'].value_counts()
        st.bar_chart(sentiment_counts)
