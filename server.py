# server.py
import os
import sys
from flask import Flask, render_template, request
from Programs import Program
import pickle
from compare_to_user import generate_recommendation
import time
import json


# Create Flask app
app = Flask(__name__)
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True #make sure templates are reloaded when changed
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0 #disable caching of static files

# Define a simple route
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/survey', methods=['GET'])
def start_survey():
    return render_template('survey.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html')

@app.route('/my-story', methods=['GET'])
def my_story():
    return render_template('my-story.html')

@app.route('/brain-retraining', methods=['GET'])
def what_is_brain_retraining():
    return render_template('brain-retraining.html')

@app.route('/recommendations', methods=['POST'])
def recommendations():
    # Get the form data
    q1 = request.form.get('question1')
    a1 = request.form.get('answer1')
    q2 = request.form.get('question2')
    a2 = request.form.get('answer2')
    q3 = request.form.get('question3')
    a3 = request.form.get('answer3')
    answers = [a1,a2,a3]
    questions = [q1,q2,q3]

    max_price = request.form.get('max-price')
    refund = 'refund' in request.form
    financial_aid = 'financial-aid' in request.form
    coaching = 'coaching' in request.form
    community = 'community' in request.form
    forum = 'forum' in request.form

    # Open list of programs
    with open('program_list.pkl', 'rb') as file:
        program_list = pickle.load(file)
    N_progs = len(program_list)

    # Generate the ranked list of programs, regardless of filter settings
    summary, ranking_idx = generate_recommendation(answers, questions)

    # Create a list of ranked programs with their data
    ranked_programs = []
    for idx in ranking_idx:
        program = program_list[idx]
        ranked_programs.append({
            'name': program.name,
            'score': int(100*summary['Overall Score'][idx]),
            'url': program.url,
            'price': program.price,
            'pricing_notes': program.pricing_notes,
            'refund_policy': program.refundable,
            'coaching': program.coaching,
            'forum': program.forum,
            'community': program.community,
            'financial_aid': program.discount,
            'description': program.description if hasattr(program, 'description') else '',
            'user_sentences': summary['User Semantic Sentences'][idx][:3], #to display the top 3 matching sentences
            'program_sentences': summary['Semantic Sentences'][idx][:3],
            'user_tone_sentences': summary['User Tone Sentences'][idx][:3],
            'program_tone_sentences': summary['Tone Sentences'][idx][:3],
        })

    ranked_programs_json = json.dumps(ranked_programs)

    # Build and return the HTML with the recommendations
    return render_template('recommendations.html', 
                          ranked_programs=ranked_programs_json,
                          max_price_survey = max_price,
                          refund_survey=refund,
                          financial_aid_survey = financial_aid,
                          coaching_survey=coaching,
                          community_survey=community,
                          forum_survey=forum,
                          version=time.time())


# Start the server
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(port=port)