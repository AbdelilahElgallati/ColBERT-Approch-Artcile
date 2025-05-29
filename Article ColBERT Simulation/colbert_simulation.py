import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import nltk
from nltk.tokenize import sent_tokenize
import matplotlib.pyplot as plt
import torch.nn as nn
import sys
import networkx as nx
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches

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
        
    def create_architecture_graph(self):
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes for each component
        nodes = [
            ('input', 'Input Text'),
            ('tokenizer', 'Tokenizer'),
            ('bert', 'BERT Encoder'),
            ('sentence_path1', 'Sentence Path 1'),
            ('sentence_path2', 'Sentence Path 2'),
            ('text_path', 'Text Path'),
            ('concat', 'Concatenation'),
            ('final', 'Final Layers'),
            ('output', 'Output')
        ]
        
        # Add nodes with positions
        pos = {
            'input': (0, 0),
            'tokenizer': (2, 0),
            'bert': (4, 0),
            'sentence_path1': (6, 1),
            'sentence_path2': (6, -1),
            'text_path': (6, 0),
            'concat': (8, 0),
            'final': (10, 0),
            'output': (12, 0)
        }
        
        # Add edges
        edges = [
            ('input', 'tokenizer'),
            ('tokenizer', 'bert'),
            ('bert', 'sentence_path1'),
            ('bert', 'sentence_path2'),
            ('bert', 'text_path'),
            ('sentence_path1', 'concat'),
            ('sentence_path2', 'concat'),
            ('text_path', 'concat'),
            ('concat', 'final'),
            ('final', 'output')
        ]
        
        # Add nodes and edges
        for node, label in nodes:
            G.add_node(node, label=label)
        G.add_edges_from(edges)
        
        return G, pos

    def visualize_architecture(self, current_step=None):
        G, pos = self.create_architecture_graph()
        
        plt.figure(figsize=(15, 8))
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                             node_size=2000, alpha=0.7)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, edge_color='gray', 
                             arrows=True, arrowsize=20)
        
        # Draw labels
        labels = {node: data['label'] for node, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8)
        
        # Highlight current step if provided
        if current_step:
            nx.draw_networkx_nodes(G, pos, nodelist=[current_step],
                                 node_color='red', node_size=2000, alpha=0.7)
        
        plt.title("ColBERT Architecture Visualization")
        plt.axis('off')
        plt.tight_layout()
        
        if current_step:
            plt.savefig(f'colbert_step_{current_step}.png')
        else:
            plt.savefig('colbert_architecture.png')
        plt.close()

    def process_text(self, text):
        try:
            # Visualize initial architecture
            self.visualize_architecture()
            
            # Step 1: Split text into sentences
            self.visualize_architecture('input')
            sentences = sent_tokenize(text)
            print("\n1. Text split into sentences:")
            for i, sent in enumerate(sentences, 1):
                print(f"Sentence {i}: {sent}")
            
            # Step 2: Tokenize each sentence
            self.visualize_architecture('tokenizer')
            tokenized_sentences = []
            for sent in sentences:
                tokens = self.tokenizer.tokenize(sent)
                tokenized_sentences.append(tokens)
                print(f"\n2. Tokenized sentence: {' '.join(tokens)}")
            
            # Step 3: Generate embeddings
            self.visualize_architecture('bert')
            print("\n3. Generating embeddings...")
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
            
            # Step 4: Process through parallel neural network
            self.visualize_architecture('sentence_path1')
            print("\n4. Processing through parallel neural network...")
            with torch.no_grad():
                output, s1_features, s2_features, text_features = self.pnn(
                    sentence_embeddings[0].unsqueeze(0),
                    sentence_embeddings[1].unsqueeze(0),
                    whole_text_embedding.unsqueeze(0)
                )
            
            # Step 5: Final processing
            self.visualize_architecture('output')
            print("\n5. Results:")
            print(f"   Humor probability: {output[0].item():.2f}")
            print(f"   Number of sentences: {len(sentences)}")
            print(f"   Embedding dimensions: {sentence_embeddings[0].shape}")
            
            return sentences, tokenized_sentences, sentence_embeddings, whole_text_embedding, output
            
        except Exception as e:
            print(f"Error processing text: {e}")
            sys.exit(1)

def main():
    try:
        # Create simulation instance
        sim = ColbertSimulation()
        
        # Example text
        text = "Is the doctor at home? No, come right in."
        
        print("\nOriginal text:", text)
        
        # Process the text
        sentences, tokenized_sentences, sentence_embeddings, whole_text_embedding, output = sim.process_text(text)
        
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 