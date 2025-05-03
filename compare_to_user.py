#!/usr/bin/env python
# coding: utf-8

# In[275]:


from Programs import Program, get_semantic_embedding, get_tone_embedding
from WebScrapeText import split_sent
import pickle
import numpy as np
import tensorflow as tf
import time


# In[252]:


with open('program_list.pkl', 'rb') as file:
    program_list = pickle.load(file)

N_progs = len(program_list)
N_similarities = 5 #How many similarity scores to use in averaging

octavia_spencer_answers = [
    "I’m proudest of the fact that I didn’t give up. This industry can be tough—life can be even tougher—but I stayed the course, trusted God, and trusted myself. I’m proud that I’ve been able to tell stories that matter, stories that shine a light on the overlooked and the underestimated. And I’m proud that I can be a source of encouragement to anyone who’s ever felt unseen.",
    
    "Integrity, empathy, and faith. I try to approach everything I do with a heart of compassion and a sense of responsibility. I ask myself: Is this true to who I am? Will it help somebody? Will I be proud of this later? If the answer is yes, I step forward with confidence.",
    
    "I admire people who are kind even when no one’s watching. People who listen more than they talk, and who act with grace under pressure. I admire quiet strength. The kind that doesn’t have to prove itself—it just shows up, does the work, and uplifts everyone in the room.",
    
    "To love, to serve, and to grow. I believe we’re all here to make this world a little brighter for each other. Life isn’t about perfection—it’s about progress. About showing up with your whole heart, learning from your mistakes, and doing what you can to leave a positive mark.",
    
    "I remember sitting around the table with my family, eating, laughing, telling stories, and just feeling that warmth. We didn’t have everything, but we had each other—and that was enough. That sense of belonging, of being seen and loved, is something I carry with me always.",
    
    "With humility. Growth requires you to admit you don’t know everything—and that’s okay. I read, I listen, I surround myself with people who challenge and support me. I take time to reflect. And I remind myself that becoming your best self isn’t a destination—it’s a lifelong journey.",
    
    "Happiness, to me, is peace. It’s being able to look in the mirror and feel proud of who you are—not just what you’ve done. It’s laughter with friends, a quiet moment in the middle of chaos, or using your voice to uplift someone else. It’s knowing that you’re aligned with your purpose.",
    
    "Life has taught me that your worth isn’t defined by anyone else’s opinion. That setbacks can be setups for something greater. That kindness isn’t weakness. That standing in your truth may not always be easy, but it’s always worth it. And above all, that we’re not here to compete—we’re here to connect."
]
octavia_spencer_qs = [0,1,2,3,4,5,6,7]


# In[385]:


def split_user_text(answers, q_list):
    """Split user text into strings of sentences, create array that indexes each sentence with the question it came from"""
    q_indx = [] 
    user_text = []
    for i in range(len(answers)):
        ans_split = split_sent(answers[i])
        q_indx.extend([q_list[i]]*(len(ans_split))) #indexes each string with the question it came from
        user_text.extend(ans_split)
    return user_text, q_indx


# In[281]:


def compare_user_prog(user_text):
    """Generate embeddings for user text, compare to program text, return scores and indices for the top 5 program sentence matches for each program for each user sentence."""
    user_sem_embeddings = [get_semantic_embedding(text) for text in user_text]
    user_tone_embeddings = [get_tone_embedding(text) for text in user_text]
    print(time.time())
    N_user_embeddings = len(user_sem_embeddings)
    top_sem_similarities_indxs = np.zeros((N_progs,N_user_embeddings,N_similarities)) # index of the top N most similar program sentences to each user sentence
    top_sem_similarities_scores = np.zeros((N_progs,N_user_embeddings,N_similarities))# score of the top N most similar program sentences to each user sentence
    top_tone_similarities_indxs = np.zeros((N_progs,N_user_embeddings,N_similarities))# index of the top N most similar program sentences to each user sentence
    top_tone_similarities_scores = np.zeros((N_progs,N_user_embeddings,N_similarities))# score of the top N most similar program sentences to each user sentence
    for p in range(N_progs):
        prog = program_list[p]
        for i in range(N_user_embeddings): #for the ith user sentence
            prog_sem_similarities = np.zeros(len(prog.text))
            prog_tone_similarities = np.zeros(len(prog.text))
            for j in range(len(prog.text)): #for the jth program sentence
                prog_sem_similarities[j] = -tf.keras.losses.cosine_similarity(
                    user_sem_embeddings[i],prog.semantic_embeddings[j]) #calculate similarity (flip the sign bec it's defined a negative for use as a loss in tensorflow
                prog_tone_similarities[j] = -tf.keras.losses.cosine_similarity(
                    user_tone_embeddings[i],prog.tone_embeddings[j])
            indices_sem = tf.argsort(prog_sem_similarities, direction = 'DESCENDING') #sort by similarity
            indices_tone = tf.argsort(prog_tone_similarities, direction = 'DESCENDING') #sort by similarity
            top_sem_similarities_indxs[p,i,:] = indices_sem[:N_similarities] #store only the top N most similar program sentences
            top_sem_similarities_scores[p,i,:] = [prog_sem_similarities[indx] for indx in indices_sem[:N_similarities]] #pull out the scores for the top N most similar
            top_tone_similarities_indxs[p,i,:] = indices_tone[:N_similarities] #store only the top N most similar
            top_tone_similarities_scores[p,i,:] = [prog_tone_similarities[indx] for indx in indices_tone[:N_similarities]] #pull out the scores for the top N most similar
    print(time.time())
    return top_sem_similarities_indxs, top_sem_similarities_scores,top_tone_similarities_indxs, top_tone_similarities_scores



def rank_progs(UserText, q_indx, topsem_indx, topsem_score, toptone_indx, toptone_score):
    """Calculate a score for each program by averaging the tone and semantic scores for its top 10 matching sentences, rank programs accordingly"""
    summary = {'Name': [], 'Program Index': [],
               'Semantic Sentences': [], 'User Semantic Sentences':[], 'Semantic Questions':[], 'Semantic Scores':[], 'Avg Semantic Score': [],
               'Tone Sentences':[], 'User Tone Sentences':[], 'Tone Questions':[], 'Tone Scores':[], 'Avg Tone Score': [],
               'Overall Score':[]}
    for p in range(N_progs): #for each program
        summary['Name'].append(program_list[p].name)
        
        flat_sem_scores = topsem_score[p].flatten() #flatten the scores for sorting
        flat_sem_indx = topsem_indx[p].flatten() #flatten the sentence indices for reference after sorting
        flat_tone_scores = toptone_score[p].flatten()
        flat_tone_indx = toptone_indx[p].flatten()
        
        Topsem_indx = tf.argsort(flat_sem_scores,direction='DESCENDING')[:10].numpy() #get indices of the top ten scores
        topsem_scores = [flat_sem_scores[indx] for indx in Topsem_indx] #get the top ten scores
        topsem_sent_indx = [int(flat_sem_indx[indx]) for indx in Topsem_indx] #get the indices for the top ten sentences
        sem_sentences = [program_list[p].text[indx] for indx in topsem_sent_indx] #get the top ten sentences
        user_sem_indx = np.floor([indx/N_similarities for indx in Topsem_indx]).astype(int)
        user_sem_sentences = [UserText[indx] for indx in user_sem_indx]
        q_sem_indx = [q_indx[indx] for indx in user_sem_indx]
        
        Toptone_indx = tf.argsort(flat_tone_scores,direction='DESCENDING')[:10].numpy()
        toptone_scores = [flat_tone_scores[indx] for indx in Toptone_indx] #get the top ten scores
        toptone_sent_indx = [int(flat_tone_indx[indx]) for indx in Toptone_indx] #get the indices for the top ten sentences
        tone_sentences = [program_list[p].text[indx] for indx in toptone_sent_indx] #get the top ten sentences
        user_tone_indx = np.floor([indx/N_similarities for indx in Toptone_indx]).astype(int)
        user_tone_sentences = [UserText[indx] for indx in user_tone_indx]
        q_tone_indx = [q_indx[indx] for indx in user_tone_indx]
        
        overall_score = (0.5)*np.mean(topsem_scores) + (0.5)*np.mean(toptone_scores) #for now, equal weights to semantic and tone scores
        
        summary['Semantic Sentences'].append(sem_sentences)
        summary['User Semantic Sentences'].append(user_sem_sentences)
        summary['Semantic Questions'].append(q_sem_indx)
        summary['Semantic Scores'].append(topsem_scores)
        summary['Avg Semantic Score'].append(np.mean(topsem_scores))
        
        summary['Tone Sentences'].append(tone_sentences)
        summary['User Tone Sentences'].append(user_tone_sentences)
        summary['Tone Questions'].append(q_tone_indx)
        summary['Tone Scores'].append(toptone_scores)
        summary['Avg Tone Score'].append(np.mean(toptone_scores))
        
        summary['Overall Score'].append(overall_score)
        
    top_indx = tf.argsort(summary['Overall Score'],direction='DESCENDING')
    top_indx = top_indx._numpy()

    return summary, top_indx


def generate_recommendation(user_text,q_list):
    
    UserText, q_indx = split_user_text(user_text,q_list)
    sem_indx, sem_scores, tone_indx, tone_scores = compare_user_prog(UserText)
    summary, top_indx = rank_progs(UserText, q_indx, sem_indx,sem_scores, tone_indx, tone_scores)
    
    return summary, top_indx

    
# for i in top_indx[:3]:
#     print(f"{summary['Name'][i]}, (Overall Score {summary['Overall Score'][i]:.4f})")
#     print("-"*50)
#     print(f"Average Semantic Score: {summary['Avg Semantic Score'][i]:.4f}")
#     for j in range(len(summary['Semantic Scores'][i])):
#         print(f"PROGRAM SENTENCE: {summary['Semantic Sentences'][i][j]}")
#         print(f"USER SENTENCE: {summary['User Semantic Sentences'][i][j]}")
#         print(f"SEMANTIC SCORE: {summary['Semantic Scores'][i][j]:.4f}")
#     print(f"\nAverage Tone Score: {summary['Avg Tone Score'][i]:.4f}")
#     for j in range(len(summary['Tone Scores'][i])):
#         print(f"PROGRAM SENTENCE: {summary['Tone Sentences'][i][j]}")
#         print(f"USER SENTENCE: {summary['User Tone Sentences'][i][j]}")
#         print(f"TONE SCORE: {summary['Tone Scores'][i][j]:.4f}")
#     print('\n\n')
    
