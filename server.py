# server.py
import os
import sys
from flask import Flask, render_template, request, jsonify
import requests
from deploy_setup import download_models, init_db, log_survey_data, get_db_connection
from dotenv import load_dotenv
import uuid

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

try:
    with open('program_list.pkl', 'rb') as file:
        program_list = pickle.load(file)
    N_progs = len(program_list)
    prices = [program_list[i].price for i in range(N_progs)]
except Exception as e:
    print(f"Error loading program list: {e}")
    program_list = []
    prices = []
    N_progs = 0

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/survey', methods=['GET'])
def start_survey():
    if not program_list:
        return render_template('maintenance.html'), 503
    
    refund_data = []
    for program in program_list:
        if hasattr(program, 'refund_term') and program.refund_term:
            refund_data.append(program.refund_term)
        else:
            refund_data.append([])

    access_length_data = []
    for program in program_list:
        if hasattr(program, 'access_length') and program.access_length:
            access_length_data.append(program.access_length)
        else:
            access_length_data.append([])

    return render_template('survey.html', price_list=prices, refund_data=refund_data, access_length_data=access_length_data)

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
    if not program_list:
        return render_template('maintenance.html'), 503
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
    refund_within_7 = 'refund-within-7' in request.form
    refund_within_14 = 'refund-within-14' in request.form
    refund_within_30 = 'refund-within-30' in request.form
    refund_after_6 = 'refund-after-6' in request.form
    refund_6_12 = 'refund-after-6-before-12' in request.form
    refund_any_time = 'refund-any-time' in request.form
    refund_free_trial = 'refund-free-trial' in request.form

    access_length = 'access-length' in request.form
    access_monthly = 'access-monthly' in request.form
    access_multi_month = 'access-multi-month' in request.form
    access_12_months = 'access-12-months' in request.form
    access_24_months = 'access-24-months' in request.form
    access_lifetime = 'access-lifetime' in request.form

    financial_aid = 'financial-aid' in request.form
    coaching = 'coaching' in request.form
    community = 'community' in request.form
    forum = 'forum' in request.form
    store_answers_consent = 'store-answers' in request.form


    session_id = str(uuid.uuid4()) #create an anonymous session id

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
            'access_length': program.access_length,
            'refund_policy': program.refundable,
            'refund_terms': program.refund_term,
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
            #Also store top sentences used for each program, the ones that are used for the score (I believe that's 5 from semantics and 10 from tone)
        }
        for i in range(min(5, len(ranked_programs)))
    ]

    top_prog_sentences = [
        {
            'name': summary['Name'][idx],
            'sem_sentences': summary['Semantic Sentences'][idx],
            'tone_sentences': summary['Tone Sentences'][idx]
        }
        for idx in ranking_idx
    ]

    if store_answers_consent:
        user_sentences = [
            {
                'answers': answers,
                'user_sem_sentences': summary['User Semantic Sentences'][idx],
                'user_tone_sentences': summary['User Tone Sentences'][idx],
            }
            for idx in ranking_idx
        ]
    else:
        user_sentences = None

    initial_filters = {
        'max_price': max_price,
        'refund': refund,
        'refund_7': refund_within_7,
        'refund_14': refund_within_14,
        'refund_30': refund_within_30,
        'refund_6_12': refund_6_12,
        'refund_6': refund_after_6,
        'refund_anylength': refund_any_time,
        'access_length': access_length,
        'access_monthly': access_monthly,
        'access_multi_month': access_multi_month,
        'access_12_months': access_12_months,
        'access_24_months': access_24_months,
        'access_lifetime': access_lifetime,
        'free_trial': refund_free_trial,
        'financial_aid': financial_aid,
        'coaching': coaching,
        'community': community,
        'forum': forum
    }

    log_survey_data(session_id, questions, user_sentences, top_prog_sentences, top_5_programs, initial_filters)

    ranked_programs_json = json.dumps(ranked_programs)

    # Build and return the HTML with the recommendations
    return render_template('recommendations.html', 
                          ranked_programs=ranked_programs_json,
                          price_list = prices,
                          max_price_survey = max_price,
                          refund_survey=refund,
                          refund_within_7 = refund_within_7,
                          refund_within_14 = refund_within_14,
                          refund_within_30 = refund_within_30,
                          refund_6_12 = refund_6_12,
                          refund_after_6 = refund_after_6,
                          refund_any_time = refund_any_time,
                          free_trial = refund_free_trial,
                          access_length_survey=access_length,
                          access_monthly=access_monthly,
                          access_multi_month=access_multi_month,
                          access_12_months=access_12_months,
                          access_24_months=access_24_months,
                          access_lifetime=access_lifetime,
                          financial_aid_survey = financial_aid,
                          coaching_survey=coaching,
                          community_survey=community,
                          forum_survey=forum,
                          session_id = session_id,
                          version=time.time())

#track which program websites get visited
@app.route('/track-click', methods=['POST'])
def track_click():
    conn = None
    cur = None
    try:
        data = request.get_json()
        program_name = data.get('program_name')
        program_url = data.get('program_url')
        session_id = data.get('session_id')
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute('''
            INSERT INTO click_tracking (session_id, program_name, program_url, timestamp)
            VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
        ''', (session_id, program_name, program_url))
        
        conn.commit()
        
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error tracking click: {e}")
        return jsonify({'success': False}), 500
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()
    
#track filter usage
@app.route('/log-filter-usage', methods=['POST'])
def log_filter_usage():
    cur = None
    conn = None
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        filter_settings = data.get('filter_settings')
        results_count = data.get('results_count')
        displayed_programs = data.get('displayed_programs')
        
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute('''
            INSERT INTO filter_usage (session_id, filter_settings, results_count, displayed_programs)
            VALUES (%s, %s, %s, %s)
        ''', (session_id, json.dumps(filter_settings), results_count, json.dumps(displayed_programs)))
        
        conn.commit()        
        return jsonify({'success': True})
    except Exception as e:
        print(f"Error logging filter usage: {e}")
        return jsonify({'success': False}), 500
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

# Start the server
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7000))
    app.run(host='0.0.0.0', port=port)