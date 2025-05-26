# Brand Sentiment Analyzer 🚀

This project is a **BERT-powered sentiment analysis web app** for analyzing brand reviews and customer feedback.

## Features
✅ Predict sentiment (Positive, Negative, Uncertain) for single text input.  
✅ Upload CSV files and get batch predictions with confidence scores.  
✅ Download analyzed results as CSV.  
✅ Built using **Streamlit**, **Transformers**, and **PyTorch**.  
✅ Deployed on **Streamlit Cloud** for easy access.

## How It Works
- Fine-tuned `bert-base-uncased` model on IMDB data for binary sentiment classification.
- Confidence thresholding ensures reliable predictions.
- The app processes text inputs or CSV files and outputs sentiment predictions.

## Future Work
- Fine-tune BERT on **brand-specific datasets** for improved accuracy.
- Expand to **multimodal sentiment analysis** (text + image).
- Deploy as a **full-scale web service** for brands.

## Run Locally
```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
