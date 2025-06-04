#Define the Program class
import json
import nltk
class Program:
    """Class to hold information about each program"""
    def __init__(self,name,description = [],url = [],price = [],pricing_notes = [],discount = [],refundable = [],coaching = [],community = [],forum = [],text = []):
        self.name = name
        self.description = description
        self.url = url
        self.price = price
        self.pricing_notes = pricing_notes
        self.discount = discount
        self.coaching = coaching
        self.refundable = refundable
        self.community = community
        self.forum = forum
        self.text = [text]
        self.semantic_embeddings = []
        self.tone_embeddings = []

    def add_text(self,new_text):
        self.text.append(new_text)

    def add_semantic_embedding(self,embedding):
        self.semantic_embeddings.append(embedding)

    def add_tone_embedding(self,embedding):
        self.tone_embeddings.append(embedding)

    def to_json(self):
        return {"name": self.name, "description": self.description, "url": self.url, "price": self.price, "pricing_notes": self.pricing_notes,
               "discount": self.discount, "coaching": self.coaching, "refundable": self.refundable, "community": self.community, "forum": self.forum,
               "text": self.text}
    def to_json_embed(self):
        return {"semantic_embeddings": self.semantic_embeddings, "tone_embeddings": self.tone_embeddings}
    
    @classmethod
    def from_json(cls, data):
        return cls(data["name"], data["description"], data["url"], data["price"], data["pricing_notes"], data["discount"], data["coaching"], 
                   data["refundable"], data["community"],data["forum"],data["text"])

# Define Functions for generating embeddings
# Import packages for text embeddings
# For semantic content embedding using mpnet-base-v2
from sentence_transformers import SentenceTransformer
# For tone/emotion embedding using twitter-roberta-base-emotion
from transformers import AutoTokenizer
import torch
import numpy as np
from optimum.onnxruntime import ORTModelForFeatureExtraction


# Load the MPNet model for content understanding
semantic_model = ORTModelForFeatureExtraction.from_pretrained("onnx/mpnet_quantized")
semantic_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
# Load the RoBERTa model for tone detection
tone_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
tone_model = ORTModelForFeatureExtraction.from_pretrained("onnx/emotion_quantized")

def get_semantic_embedding(text):
    inputs = semantic_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        output = semantic_model(**inputs).last_hidden_state  # (batch, seq_len, hidden_dim)
        # Use attention mask for proper mean pooling
        attention_mask = inputs['attention_mask']
        mask_expanded = attention_mask.unsqueeze(-1).expand(output.size()).float() #expand the mask to hit each unit in each token
        sum_embeddings = torch.sum(output * mask_expanded, dim=1) #this zeros out the padding tokens
        sum_mask = mask_expanded.sum(dim=1) #Only want to divide by the number of real tokens, ignoring padding tokens
        embedding = sum_embeddings / sum_mask
        return embedding.squeeze()

# Example usage for tone embeddings
def get_tone_embedding(text):
    inputs = tone_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = tone_model(**inputs) #**inputs upacks the dictionary that tone_tokenizer creates into keyword arguments, 
        #e.g., tone_model(**inputs) ==  tone_model(input_ids=tensor([[101, 2023, 5628, 3437, 102]]), attention_mask=tensor([[1, 1, 1, 1, 1]]))

    # Get last hidden state
    last_hidden_state = outputs.last_hidden_state
    attention_mask = inputs['attention_mask']
    # Only consider non-padded tokens
    seq_length = attention_mask.sum(dim=1).item()  # Actual sequence length
    
    #we're going to take a weighted average of the last hidden layer, 50% to the CLS token, and the last 50% distributed among the rest
    # Create weights only for actual tokens (not padding)
    weights = np.zeros(seq_length)
    weights[0] = 0.5  # CLS token gets 50%
    if seq_length > 1:
        weights[1:] = 1/(2*(seq_length-1))  # Remaining tokens share the other 50%
    full_weights = torch.zeros(last_hidden_state.shape[1])
    full_weights[:seq_length] = torch.tensor(weights, dtype=full_weights.dtype)
    
    # Apply attention mask to weights (zero out padding positions)
    masked_weights = full_weights * attention_mask.squeeze().float()
    
    # Reshape for broadcasting and compute weighted average
    weights_tensor = masked_weights.unsqueeze(0).unsqueeze(-1)
    avg_embedding = (last_hidden_state * weights_tensor).sum(dim=1).squeeze()
    
    return avg_embedding.detach().numpy()


def split_sent(text):
    sentences = nltk.sent_tokenize(text)
    return sentences