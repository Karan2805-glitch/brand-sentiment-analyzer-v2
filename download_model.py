from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load model from Hugging Face Hub
model = AutoModelForSequenceClassification.from_pretrained("Karan2805-glitch/brand-sentiment-bert")
tokenizer = AutoTokenizer.from_pretrained("Karan2805-glitch/brand-sentiment-bert")

# Save locally
model.save_pretrained("./model")
tokenizer.save_pretrained("./model")
