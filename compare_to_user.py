#!/usr/bin/env python
# coding: utf-8



from Programs import Program, get_semantic_embedding, get_tone_embedding, split_sent
import pickle
import numpy as np
import time




with open('program_list.pkl', 'rb') as file:
    program_list = pickle.load(file)

N_progs = len(program_list)
N_similarities = 5 #How many similarity scores to use in averaging


def split_user_text(answers, q_list):
    """Split user text into strings of sentences, create array that indexes each sentence with the question it came from"""
    q_indx = [] 
    user_text = []
    for i in range(len(answers)):
        ans_split = split_sent(answers[i])
        q_indx.extend([q_list[i]]*(len(ans_split))) #indexes each string with the question it came from
        user_text.extend(ans_split)
    return user_text, q_indx

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

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
                prog_sem_similarities[j] = cosine_sim(user_sem_embeddings[i],prog.semantic_embeddings[j]) #calculate similarity (flip the sign bec it's defined a negative for use as a loss in tensorflow
                prog_tone_similarities[j] = cosine_sim(user_tone_embeddings[i],prog.tone_embeddings[j])
            indices_sem = np.argsort(prog_sem_similarities)[::-1] #sort by similarity
            indices_tone = np.argsort(prog_tone_similarities)[::-1] #sort by similarity
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
        
        Topsem_indx = np.argsort(flat_sem_scores)[::-1][:10] #get indices of the top ten scores
        topsem_scores = [flat_sem_scores[indx] for indx in Topsem_indx] #get the top ten scores
        topsem_sent_indx = [int(flat_sem_indx[indx]) for indx in Topsem_indx] #get the indices for the top ten sentences
        sem_sentences = [program_list[p].text[indx] for indx in topsem_sent_indx] #get the top ten sentences
        user_sem_indx = np.floor([indx/N_similarities for indx in Topsem_indx]).astype(int)
        user_sem_sentences = [UserText[indx] for indx in user_sem_indx]
        q_sem_indx = [q_indx[indx] for indx in user_sem_indx]
        
        Toptone_indx = np.argsort(flat_tone_scores)[::-1][:10]
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
        
    top_indx = np.argsort(summary['Overall Score'])[::-1]

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
    
