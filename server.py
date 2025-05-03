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
    return 'Start survey'

# Start the server
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(port=port)