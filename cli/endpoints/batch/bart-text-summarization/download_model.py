from transformers import pipeline

model_local_path = 'model'

summarizer = pipeline("summarization", model="facebook/bart-large-cnn", framework="pt")
summarizer.save_pretrained(model_local_path)