import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import pandas as pd

# 🌟 Streamlit Page Config
st.set_page_config(page_title="Brand Sentiment Analyzer", page_icon="🧠", layout="centered")
st.title("🧠 Brand Sentiment Analyzer")
st.markdown("""
Welcome to the **Brand Sentiment Analyzer**!  
This app uses a **BERT model** fine-tuned on IMDB movie reviews to predict sentiment as **Positive**, **Negative**, or **Uncertain**.  
Upload your data or type in your own text to get started! 🚀
""")

# 🌟 Load Model and Tokenizer from Hugging Face Hub (default to CPU)
model_path = "Karan2805-glitch/brand-sentiment-bert"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 🌟 Prediction Function
def predict_sentiment(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to("cpu") for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        confidence, prediction = torch.max(probs, dim=-1)
        confidence = confidence.item()
        prediction = prediction.item()
    
    if confidence < 0.6:
        return "🤔 Uncertain", confidence
    else:
        return ("✅ Positive" if prediction == 1 else "❌ Negative"), confidence

# 🌟 Single Text Prediction
st.subheader("🎯 Single Text Sentiment Prediction")
user_input = st.text_input("Enter text for sentiment prediction:")

if user_input:
    prediction, confidence = predict_sentiment(user_input)
    st.success(f"Prediction: {prediction} (Confidence: {confidence:.2f})")

# 🌟 Batch CSV Upload
st.subheader("📂 Analyze a File (CSV Upload)")
uploaded_file = st.file_uploader("Upload a CSV file with a 'text' column", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("📊 Sample Data", df.head())

    if 'text' not in df.columns:
        st.error("❗ CSV must have a 'text' column!")
    else:
        st.info("⏳ Running sentiment analysis...")
        sentiments = []
        confidences = []
        for text in df['text']:
            if pd.isnull(text):
                sentiments.append("Unknown")
                confidences.append(0)
            else:
                sentiment, confidence = predict_sentiment(str(text))
                sentiments.append(sentiment)
                confidences.append(round(confidence, 2))

        df['Sentiment'] = sentiments
        df['Confidence'] = confidences
        st.write("✅ Analysis Results", df)

        # 📈 Bar Chart
        sentiment_counts = df['Sentiment'].value_counts()
        st.bar_chart(sentiment_counts)

        # 💾 Download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Analyzed Data", csv, "analyzed_sentiment.csv", "text/csv")

st.markdown("---")
st.markdown("Made with ❤️ using BERT, Streamlit, and Hugging Face by Karan2805-glitch.")
