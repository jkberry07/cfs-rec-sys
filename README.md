Recommender system for me/cfs recovery programs. It compares user text to program text by calculating two different embeddings for each sentence, 
one for semantic meaning (using a quantized version of mpnet) and one for tone (using a quantized version of twitter-roberta-base-emotion).
The code calculates cosine similarity scores for semantics and tone, uses the highest scores to calculate an overall score for each program, and 
returns a ranked list of programs.

Also included is the server code and html pages for the associated website: 
