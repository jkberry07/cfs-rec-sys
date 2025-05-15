# server.py
import os
import sys
from flask import Flask, render_template, request
# from Programs import Program
# import pickle


# Create Flask app
app = Flask(__name__)

# Define a simple route
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/start_survey', methods=['GET'])
def start_survey():
    return render_template('start-survey.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET'])
def contact():
    return render_template('contact.html')

@app.route('/my-story', methods=['GET'])
def my_story():
    return render_template('my_story.html')

@app.route('/recommendations', methods=['POST'])
def recommendations():
    # Get the form data
    # Generate recommendations based on the form data
    # Build and return the HTML with the recommendations
    return render_template('recommendations.html')

# Start the server
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(port=port)