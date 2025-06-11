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
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Create surveys table if it doesn't exist
        cur.execute('''
            CREATE TABLE IF NOT EXISTS surveys (
                id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                questions JSONB,
                top_programs JSONB
            )
        ''')
        
        conn.commit()
        cur.close()
        conn.close()
        print("Database initialized successfully")
    except Exception as e:
        print(f"Error initializing database: {e}")


#log survey data
def log_survey_data(questions, top_programs):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        cur.execute('''
            INSERT INTO surveys (questions, top_programs)
            VALUES (%s, %s)
        ''', (
            json.dumps(questions),
            json.dumps(top_programs)
        ))
        
        conn.commit()
        cur.close()
        conn.close()
        print("Survey data logged successfully")
    except Exception as e:
        print(f"Error logging survey data: {e}")