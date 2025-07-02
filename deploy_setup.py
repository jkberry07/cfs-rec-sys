#for downloading models
import os
import zipfile
import requests
#for setting up and connecting to db
import psycopg2
from urllib.parse import urlparse
import json

def download_models():
    # Check if models already exist
    emotion_model = "./onnx/emotion_quantized"
    mpnet_model = "./onnx/mpnet_quantized"
    
    if os.path.exists(emotion_model) and os.path.exists(mpnet_model):
        print("Models already exist, skipping download")
        return
    
    print("Downloading models from GitHub...")
    
    # Your release download URL
    url = "https://github.com/jkberry07/cfs-rec-sys/releases/download/models-v0.1.0/onnx-models.zip"
    try:
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        
        # Download zip
        with open("onnx-models.zip", "wb") as f:
            f.write(response.content)
        
        # Extract - this should recreate the onnx/ folder structure
        with zipfile.ZipFile("onnx-models.zip", 'r') as zip_ref:
            zip_ref.extractall("./")
        
        os.remove("onnx-models.zip")
        print("Models downloaded successfully!")
        
        # Verify the structure
        if os.path.exists(emotion_model) and os.path.exists(mpnet_model):
            print("✓ Both model folders found")
        else:
            print("⚠️ Warning: Expected folder structure not found")
            
    except Exception as e:
        print(f"Error downloading models: {e}")
        raise


#connect to db
def get_db_connection():
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set")
    
    # Parse the URL
    url = urlparse(database_url)
    
    conn = psycopg2.connect(
        host=url.hostname,
        port=url.port,
        database=url.path[1:],  # Remove leading '/'
        user=url.username,
        password=url.password,
        sslmode='require'
    )
    return conn



# Initialize database table
def init_db():
    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Create recommendations table if it doesn't exist
        # Previous table was named "surveys", stored only questions and top programs
        cur.execute('''
            CREATE TABLE IF NOT EXISTS recommendations (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                session_id TEXT,
                questions JSONB,
                user_sentences JSONB,
                top_prog_sentences JSONB,
                top_programs JSONB,
                initial_filters JSONB
            )
        ''')

        cur.execute('''
            CREATE TABLE IF NOT EXISTS click_tracking (
                id SERIAL PRIMARY KEY,
                session_id TEXT,
                program_name VARCHAR(255),
                program_url TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')

        cur.execute('''
            CREATE TABLE IF NOT EXISTS filter_usage (
                id SERIAL PRIMARY KEY,
                session_id TEXT,
                filter_settings JSONB,
                results_count INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Error initializing database: {e}")
        if conn:
            conn.rollback() #rollback on error
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()


#log survey data
def log_survey_data(session_id, questions, user_sentences, top_prog_sentences, top_programs,initial_filters):
    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute('''
            INSERT INTO recommendations (session_id, questions, user_sentences, top_prog_sentences, top_programs, initial_filters)
            VALUES (%s, %s, %s, %s, %s, %s)
        ''', (
            session_id,
            json.dumps(questions),
            json.dumps(user_sentences),
            json.dumps(top_prog_sentences),
            json.dumps(top_programs),
            json.dumps(initial_filters)
        ))
        
        conn.commit()
        print("Survey data logged successfully")
    except Exception as e:
        print(f"Error logging survey data: {e}")
        if conn:
            conn.rollback()  # Rollback on error
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()