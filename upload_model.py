from transformers import AutoModelForSequenceClassification, AutoTokenizer

# Load your model from local folder
model = AutoModelForSequenceClassification.from_pretrained("./model/saved_model")
tokenizer = AutoTokenizer.from_pretrained("./model/saved_model")

# Push to Hugging Face
model.push_to_hub("Karan2805-glitch/brand-sentiment-bert")
tokenizer.push_to_hub("Karan2805-glitch/brand-sentiment-bert")
