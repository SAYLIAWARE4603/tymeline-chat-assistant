import requests
from bs4 import BeautifulSoup
import json
import csv

# Define the URL of Tymeline
url = "https://www.tymeline.app"

# Send a request to the website
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

# Extract relevant data (e.g., product descriptions, features, etc.)
data = {}

# Example: Extracting all paragraphs
data['paragraphs'] = [p.text for p in soup.find_all('p')]

# Save the data as JSON
with open('tymeline_data.json', 'w') as f:
    json.dump(data, f, indent=4)

# Optionally, save the data as CSV
with open('tymeline_data.csv', 'w') as f:
    writer = csv.writer(f)
    for key, value in data.items():
        writer.writerow([key, value])

from transformers import GPTNeoForCausalLM, GPT2Tokenizer

# Load pre-trained model and tokenizer
model_name = "EleutherAI/gpt-neo-125M"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPTNeoForCausalLM.from_pretrained(model_name)

# Generate a response using the model
def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs['input_ids'], max_length=150)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Test the LLM
prompt = "What is Tymeline?"
print(generate_response(prompt))

from transformers import Trainer, TrainingArguments, GPT2Tokenizer

# Load the pre-trained tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the padding token to be the end-of-sequence token
tokenizer.pad_token = tokenizer.eos_token

# Create a dataset for fine-tuning
class TymelineDataset:
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        inputs = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length, return_tensors="pt")
        inputs['labels'] = inputs['input_ids']
        return inputs

# Load the data
with open('tymeline_data.json') as f:
    data = json.load(f)

texts = data['paragraphs']
dataset = TymelineDataset(texts, tokenizer)

# Fine-tuning configuration
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

# Fine-tune the model
trainer.train()

from sentence_transformers import SentenceTransformer, util
import faiss
import numpy as np

# Load a pre-trained sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Encode the Tymeline data into embeddings
corpus_embeddings = model.encode(texts, convert_to_tensor=True)

# Initialize FAISS index
index = faiss.IndexFlatL2(corpus_embeddings.shape[1])
index.add(np.array(corpus_embeddings.cpu()))

# Define a function to retrieve relevant context
def retrieve_relevant_context(query):
    query_embedding = model.encode(query, convert_to_tensor=True)
    D, I = index.search(np.array([query_embedding.cpu()]), k=5)
    return [texts[i] for i in I[0]]

# Example query
query = "Tell me about Tymeline's features"
context = retrieve_relevant_context(query)
print(context)

