# server.py
import os
import sys
from flask import Flask, render_template, request, jsonify
import requests
from deploy_setup import download_models, init_db, log_survey_data
from dotenv import load_dotenv

load_dotenv() #get environment variables on local machine.
init_db() #initialize the database

# Create Flask app
app = Flask(__name__)
app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True #make sure templates are reloaded when changed
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0 #disable caching of static files

download_models() #then ensure models are downloaded before importing anything that depends on them

import pickle
from compare_to_user import generate_recommendation
import time
import json



@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/survey', methods=['GET'])
def start_survey():
    return render_template('survey.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/program-text-sources', methods=['GET'])
def program_text_sources():
    return render_template('program-text-sources.html')

@app.route('/privacy', methods=['GET'])
def privacy():
    return render_template('privacy.html')

MAILGUN_DOMAIN = os.environ.get('MAILGUN_DOMAIN')
MAILGUN_API_KEY = os.environ.get('MAILGUN_API_KEY')
EMAIL = os.environ.get('EMAIL')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'GET':
        return render_template('contact.html')
    
    try:
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        
        # Send email via Mailgun
        response = requests.post(
            f"https://api.mailgun.net/v3/{MAILGUN_DOMAIN}/messages",
            auth=("api", MAILGUN_API_KEY),
            data={
                "from": f"Contact Form <mailgun@{MAILGUN_DOMAIN}>",
                "to": [EMAIL],
                "subject": f"Contact Form Submission from {name}",
                "text": f"""
                    New contact form submission:

                    Name: {name}
                    Email: {email}

                    Message:
                    {message}
                                    """,
                "h:Reply-To": email
            }
        )
        
        if response.status_code == 200:
            return jsonify({'success': True, 'message': 'Thank you! Your message has been sent successfully.'})
        else:
            print(f"Mailgun error: {response.status_code} - {response.text}")
            return jsonify({'success': False, 'message': 'Sorry, there was an error sending your message.'}), 500
            
    except Exception as e:
        print(f"Error sending email: {e}")
        return jsonify({'success': False, 'message': 'Sorry, there was an error sending your message.'}), 500


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
    answers = [a for a in [a1, a2, a3] if a] #filter out None if the user didn't answer all 3
    questions = [q for q in [q1, q2, q3] if q]

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
            'score': int((summary['Overall Score'][idx]-40)*2.25 + 10), #original scores typically ranged from ~40-80, rescale to range from 10-100
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
            'display_sentences': summary['Display Sentences'][idx] #For simpler display of just the top 4 program sentences, 2 for semantics, 2 for tone
        })

    # Get top 5 programs for logging
    top_5_programs = [
        {
            'name': ranked_programs[i]['name'],
            'score': ranked_programs[i]['score']
        }
        for i in range(min(5, len(ranked_programs)))
    ]

    log_survey_data(questions, top_5_programs)

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
    port = int(os.environ.get('PORT', 7000))
    app.run(host='0.0.0.0', port=port)