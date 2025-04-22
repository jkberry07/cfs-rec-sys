#Define the Program class
import json
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np
# Load the MPNet model for content understanding
semantic_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')

# Load the RoBERTa model for tone detection
tone_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-emotion")
# Load with output_hidden_states=True to access hidden representations
tone_model = AutoModelForSequenceClassification.from_pretrained(
    "cardiffnlp/twitter-roberta-base-emotion",
    output_hidden_states=True
)

def get_semantic_embedding(text):
    return semantic_model.encode(text)

# Example usage for tone embeddings
def get_tone_embedding(text):
    inputs = tone_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = tone_model(**inputs) #**inputs upacks the dictionary that tone_tokenizer creates into keyword arguments, 
        #e.g., tone_model(**inputs) ==  tone_model(input_ids=tensor([[101, 2023, 5628, 3437, 102]]), attention_mask=tensor([[1, 1, 1, 1, 1]]))

    # Get hidden states (this returns a tuple of hidden states for each layer)
    hidden_states = outputs.hidden_states
    last_hidden_state = hidden_states[-1]
    N_tokens = len(last_hidden_state[0,:,0])
    #we're going to take a weighted average of the last hidden layer, 50% to the CLS token, and the last 50% distributed among the rest
    weights = np.zeros(N_tokens)
    weights[0] = 0.5
    weights[1:] = 1/(2*(N_tokens-1))
    avg_embedding = sum(weights[i]*last_hidden_state[:,i,:] for i in range(N_tokens)).squeeze()
    return avg_embedding
