import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

# 🌟 Streamlit Page Config
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🧠", layout="centered")
st.title("🧠 Sentiment Analyzer")
st.markdown("Enter text or upload a CSV file to classify sentences as Positive, Negative, or Neutral.")

# 🌟 Load Lightweight Sentiment Model from Hugging Face
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 🌟 Mapping Model Labels
labels = {0: "❌ Negative", 1: "😐 Neutral", 2: "✅ Positive"}

# 🌟 Prediction Function
def predict_sentiment(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1).detach().cpu().numpy()
        prediction = int(probs.argmax())
    return labels[prediction]

# 🌟 Single Text Prediction
st.subheader("🎯 Single Text Sentiment Prediction")
user_input = st.text_input("Enter a sentence:")
if user_input:
    result = predict_sentiment(user_input)
    st.success(f"Sentiment: {result}")

# 🌟 CSV Upload and Batch Prediction
st.subheader("📂 Analyze a CSV File")
uploaded_file = st.file_uploader("Upload a CSV with a 'text' column", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'text' not in df.columns:
        st.error("CSV must have a 'text' column.")
    else:
        st.info("Running sentiment analysis...")
        df['Sentiment'] = df['text'].apply(lambda x: predict_sentiment(str(x)))
        st.write(df)
        st.bar_chart(df['Sentiment'].value_counts())
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results", csv, "analyzed_sentiment.csv", "text/csv")

st.markdown("---")
st.markdown("Made with ❤️ using Roberta, Streamlit, and Hugging Face.")
