#!/usr/bin/env python
# coding: utf-8



from Programs import Program, get_semantic_embedding, get_tone_embedding, split_sent
import pickle
import numpy as np
import time
from sklearn.metrics.pairwise import cosine_similarity




with open('program_list.pkl', 'rb') as file:
    program_list = pickle.load(file)

N_progs = len(program_list)
N_similarities = 5 #For each user sentence, how many similarity scores to pull out for comparison to other user sentences scores
Navg_sem = 5 #how many sentence similarities to average over when calculating the semantic scores
Navg_tone = 10 #how many sentence similarities to average over when calculating the tone scores


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
    #print(time.time())
    N_user_embeddings = len(user_sem_embeddings)
    top_sem_similarities_indxs = np.zeros((N_progs,N_user_embeddings,N_similarities)) # index of the top N most similar program sentences to each user sentence
    top_sem_similarities_scores = np.zeros((N_progs,N_user_embeddings,N_similarities))# score of the top N most similar program sentences to each user sentence
    top_tone_similarities_indxs = np.zeros((N_progs,N_user_embeddings,N_similarities))# index of the top N most similar program sentences to each user sentence
    top_tone_similarities_scores = np.zeros((N_progs,N_user_embeddings,N_similarities))# score of the top N most similar program sentences to each user sentence
    for p in range(N_progs):
        prog = program_list[p]
        prog_sem_matrix = np.array(prog.semantic_embeddings)  # (n_prog_sentences, 768)
        prog_tone_matrix = np.array(prog.tone_embeddings)    # (n_prog_sentences, 768)
        for i in range(N_user_embeddings): #for the ith user sentence
            user_sem = user_sem_embeddings[i].reshape(1, -1)
            user_tone = user_tone_embeddings[i].reshape(1, -1) #make them matrices for sklearn
            prog_sem_similarities = cosine_similarity(user_sem, prog_sem_matrix)[0]
            prog_tone_similarities = cosine_similarity(user_tone, prog_tone_matrix)[0]
            
            indices_sem = np.argsort(prog_sem_similarities)[::-1] #sort by similarity
            indices_tone = np.argsort(prog_tone_similarities)[::-1] #sort by similarity
            top_sem_similarities_indxs[p,i,:] = indices_sem[:N_similarities] #store only the top N most similar program sentences
            top_sem_similarities_scores[p,i,:] = [prog_sem_similarities[indx] for indx in indices_sem[:N_similarities]] #pull out the scores for the top N most similar
            top_tone_similarities_indxs[p,i,:] = indices_tone[:N_similarities] #store only the top N most similar
            top_tone_similarities_scores[p,i,:] = [prog_tone_similarities[indx] for indx in indices_tone[:N_similarities]] #pull out the scores for the top N most similar
    #print(time.time())
    return top_sem_similarities_indxs, top_sem_similarities_scores,top_tone_similarities_indxs, top_tone_similarities_scores



def rank_progs(UserText, q_indx, topsem_indx, topsem_score, toptone_indx, toptone_score):
    """Calculate a score for each program by averaging the tone and semantic scores for its top 10 matching sentences, rank programs accordingly"""
    summary = {'Name': [], 'Program Index': [], 'Display Sentences':[],
               'Semantic Sentences': [], 'User Semantic Sentences':[], 'Semantic Questions':[], 'Semantic Scores':[], 'Avg Semantic Score': [],
               'Tone Sentences':[], 'User Tone Sentences':[], 'Tone Questions':[], 'Tone Scores':[], 'Avg Tone Score': [],
               'Overall Score':[]}
    for p in range(N_progs): #for each program
        summary['Name'].append(program_list[p].name)
        
        flat_sem_scores = topsem_score[p].flatten() #flatten the scores for sorting (a 1D list Nuser_sentences times Nsimilarities long)
        flat_sem_indx = topsem_indx[p].flatten() #flatten the sentence indices for reference after sorting
        flat_tone_scores = toptone_score[p].flatten()
        flat_tone_indx = toptone_indx[p].flatten()
        
        Topsem_indx = np.argsort(flat_sem_scores)[::-1][:Navg_sem] #get indices of the top Navg_sem scores in the scores list
        topsem_scores = [flat_sem_scores[indx] for indx in Topsem_indx] #get the top Navg_sem scores
        topsem_sent_indx = [int(flat_sem_indx[indx]) for indx in Topsem_indx] #get the indices for the top Navg_sem sentences using score indices
        sem_sentences = [program_list[p].text[indx] for indx in topsem_sent_indx] #get the top Navg_sem sentences
        user_sem_indx = np.floor([indx/N_similarities for indx in Topsem_indx]).astype(int)
        user_sem_sentences = [UserText[indx] for indx in user_sem_indx]
        q_sem_indx = [q_indx[indx] for indx in user_sem_indx]
        
        Toptone_indx = np.argsort(flat_tone_scores)[::-1][:Navg_tone]
        toptone_scores = [flat_tone_scores[indx] for indx in Toptone_indx] #get the top Navg scores
        toptone_sent_indx = [int(flat_tone_indx[indx]) for indx in Toptone_indx] #get the indices for the top Navg sentences
        tone_sentences = [program_list[p].text[indx] for indx in toptone_sent_indx] #get the top ten sentences
        user_tone_indx = np.floor([indx/N_similarities for indx in Toptone_indx]).astype(int)
        user_tone_sentences = [UserText[indx] for indx in user_tone_indx]
        q_tone_indx = [q_indx[indx] for indx in user_tone_indx]

        #Display top 2 sentences for semantics and top 2 sentences for tone, ensuring no duplicates. Duplicates can arise if two user sentences matched best with the same program sentence
        disp_sent = [sem_sentences[0]] # Insert top semantic sentence
        sent = 1 #start comparisons with second semantic sentence
        while len(disp_sent)<2: #stop once we have two semantic sentences
            if sem_sentences[sent] not in disp_sent:
                disp_sent.append(sem_sentences[sent])
            sent = sent + 1
            if sent==len(sem_sentences): #edge case, but stop if it can't find enough unique sentences.
                break
        #Now do it for tone sentences
        sent = 0 #start with top tone sentence
        while len(disp_sent)<4: #stop once we have 4 sentences overall
            if tone_sentences[sent] not in disp_sent:
                disp_sent.append(tone_sentences[sent])
            sent = sent+1
            if sent==len(tone_sentences):
                break
        
        sem_weights = np.array([.3,.25,.2,.15,.1]) #weight the closest match the most.
        sem_score = np.sum(sem_weights*topsem_scores)
        overall_score = (0.5)*sem_score + (0.5)*np.mean(toptone_scores) #for now, equal weights to semantic and tone scores
        
        summary['Semantic Sentences'].append(sem_sentences)
        summary['User Semantic Sentences'].append(user_sem_sentences)
        summary['Semantic Questions'].append(q_sem_indx)
        summary['Semantic Scores'].append(topsem_scores)
        summary['Avg Semantic Score'].append(sem_score)
        
        summary['Tone Sentences'].append(tone_sentences)
        summary['User Tone Sentences'].append(user_tone_sentences)
        summary['Tone Questions'].append(q_tone_indx)
        summary['Tone Scores'].append(toptone_scores)
        summary['Avg Tone Score'].append(np.mean(toptone_scores))
        
        summary['Overall Score'].append(overall_score*100) #rescale as percentage
        summary['Display Sentences'].append(disp_sent)
        
    top_indx = np.argsort(summary['Overall Score'])[::-1]

    return summary, top_indx


def generate_recommendation(user_text,q_list):
    
    UserText, q_indx = split_user_text(user_text,q_list)
    sem_indx, sem_scores, tone_indx, tone_scores = compare_user_prog(UserText)
    summary, top_indx = rank_progs(UserText, q_indx, sem_indx,sem_scores, tone_indx, tone_scores)
    
    return summary, top_indx
