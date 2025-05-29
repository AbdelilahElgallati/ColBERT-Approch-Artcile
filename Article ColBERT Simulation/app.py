import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from flask import Flask, render_template, jsonify, request
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import nltk
from nltk.tokenize import sent_tokenize
import torch.nn as nn
import sys
import json

app = Flask(__name__)

def download_nltk_data():
    try:
        nltk.download('punkt', quiet=True)
    except Exception as e:
        print(f"Error downloading NLTK data: {e}")
        sys.exit(1)

# Download required NLTK data
download_nltk_data()

class ParallelNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        # Sentence path (for each sentence)
        self.sentence_path = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 20),
            nn.ReLU()
        )
        
        # Whole text path
        self.whole_text_path = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 60),
            nn.ReLU()
        )
        
        # Final layers
        self.final_layers = nn.Sequential(
            nn.Linear(100, 50),  # 100 = 20 + 20 + 60 (concatenated features)
            nn.ReLU(),
            nn.Linear(50, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )
    
    def forward(self, sentence1_emb, sentence2_emb, whole_text_emb):
        # Process each path
        s1_features = self.sentence_path(sentence1_emb)
        s2_features = self.sentence_path(sentence2_emb)
        text_features = self.whole_text_path(whole_text_emb)
        
        # Concatenate features
        combined = torch.cat([s1_features, s2_features, text_features], dim=1)
        
        # Final classification
        output = self.final_layers(combined)
        return output, s1_features, s2_features, text_features

class ColbertSimulation:
    def __init__(self):
        try:
            print("Loading BERT model and tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.model = AutoModel.from_pretrained('bert-base-uncased')
            self.pnn = ParallelNeuralNetwork()
            print("Model and tokenizer loaded successfully!")
        except Exception as e:
            print(f"Error initializing model: {e}")
            sys.exit(1)
    
    def process_text(self, text):
        steps = []
        
        # Step 1: Split text into sentences
        sentences = sent_tokenize(text)
        steps.append({
            'step': 1,
            'title': 'Text Splitting',
            'description': 'The input text is split into individual sentences.',
            'input': text,
            'output': sentences,
            'active_nodes': ['input']
        })
        
        # Step 2: Tokenize each sentence
        tokenized_sentences = []
        for sent in sentences:
            tokens = self.tokenizer.tokenize(sent)
            tokenized_sentences.append(tokens)
        
        steps.append({
            'step': 2,
            'title': 'Tokenization',
            'description': 'Each sentence is tokenized into individual words and subwords.',
            'input': sentences,
            'output': tokenized_sentences,
            'active_nodes': ['input', 'tokenizer']
        })
        
        # Step 3: Generate embeddings
        sentence_embeddings = []
        for sent in sentences:
            inputs = self.tokenizer(sent, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            sentence_embeddings.append(outputs.last_hidden_state[0].mean(dim=0))
        
        whole_text_inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            whole_text_outputs = self.model(**whole_text_inputs)
        whole_text_embedding = whole_text_outputs.last_hidden_state[0].mean(dim=0)
        
        steps.append({
            'step': 3,
            'title': 'BERT Encoding',
            'description': 'Each sentence and the whole text are encoded using BERT to generate embeddings.',
            'input': tokenized_sentences,
            'output': {
                'sentence_embeddings': [emb.tolist() for emb in sentence_embeddings],
                'whole_text_embedding': whole_text_embedding.tolist()
            },
            'active_nodes': ['input', 'tokenizer', 'bert']
        })
        
        # Step 4: Process through parallel neural network
        with torch.no_grad():
            output, s1_features, s2_features, text_features = self.pnn(
                sentence_embeddings[0].unsqueeze(0),
                sentence_embeddings[1].unsqueeze(0),
                whole_text_embedding.unsqueeze(0)
            )
        
        steps.append({
            'step': 4,
            'title': 'Parallel Processing',
            'description': 'The embeddings are processed through parallel neural networks for sentence and text paths.',
            'input': {
                'sentence_embeddings': [emb.tolist() for emb in sentence_embeddings],
                'whole_text_embedding': whole_text_embedding.tolist()
            },
            'output': {
                's1_features': s1_features[0].tolist(),
                's2_features': s2_features[0].tolist(),
                'text_features': text_features[0].tolist()
            },
            'active_nodes': ['input', 'tokenizer', 'bert', 'sentence_path1', 'sentence_path2', 'text_path']
        })
        
        # Step 5: Final processing
        steps.append({
            'step': 5,
            'title': 'Final Classification',
            'description': 'The features are concatenated and processed through final layers to get the humor probability.',
            'input': {
                's1_features': s1_features[0].tolist(),
                's2_features': s2_features[0].tolist(),
                'text_features': text_features[0].tolist()
            },
            'output': {
                'humor_probability': float(output[0].item()),
                'embedding_dimensions': sentence_embeddings[0].shape
            },
            'active_nodes': ['input', 'tokenizer', 'bert', 'sentence_path1', 'sentence_path2', 'text_path', 'concat', 'final', 'output']
        })
        
        return steps

# Initialize the simulation
sim = ColbertSimulation()

@app.route('/')
def index():
    return render_template('ColBERT.html')

@app.route('/simulation')
def simulation():
    return render_template('index.html')


@app.route('/process/<text>')
def process_text(text):
    steps = sim.process_text(text)
    return jsonify(steps)

if __name__ == '__main__':
    app.run(debug=True) 