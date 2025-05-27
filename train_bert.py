import os
os.environ["WANDB_DISABLED"] = "true"

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

# Load dataset
dataset = load_dataset("imdb")
train_data = dataset['train'].shuffle(seed=42).select(range(4000))
test_data = dataset['test'].shuffle(seed=42).select(range(1000))

# Tokenizer & model
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Tokenize
def preprocess(examples):
    return tokenizer(examples['text'], truncation=True, padding=True, max_length=512)

train_data = train_data.map(preprocess, batched=True)
test_data = test_data.map(preprocess, batched=True)
train_data = train_data.rename_column("label", "labels")
test_data = test_data.rename_column("label", "labels")
train_data.set_format("torch")
test_data.set_format("torch")

# Training setup
training_args = TrainingArguments(
    output_dir="./model/results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    logging_strategy="epoch",
    save_strategy="epoch",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
)

# Train
trainer.train()

# Save model
model.save_pretrained('./model/saved_model')
tokenizer.save_pretrained('./model/saved_model')

print("Training complete. Model saved in ./model/saved_model")
