from sqlalchemy import create_engine, text
import pickle
from Programs import Program
import os
import pandas as pd
import time

def get_sqlalchemy_engine():
    database_url = os.environ.get('DATABASE_URL')
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is not set")
    
    # SQLAlchemy engine
    engine = create_engine(database_url)
    return engine

def extract_from_db(date_time):
    engine = get_sqlalchemy_engine()
    recomm_query = "SELECT * FROM recommendations WHERE timestamp > %s"
    clicks_query = "SELECT * FROM click_tracking WHERE timestamp > %s"
    filters_query = "SELECT * FROM filter_usage WHERE timestamp > %s"
    recommendations_df = pd.read_sql_query(recomm_query, engine, params=[date_time])
    clicks_df = pd.read_sql_query(clicks_query, engine, params=[date_time])
    filters_df = pd.read_sql_query(filters_query, engine, params=[date_time])
    return recommendations_df, clicks_df, filters_df

def clean_and_aggregate(recommendations_df, clicks_df, filters_df):
    try:
        with open('program_list.pkl', 'rb') as file:
            program_list = pickle.load(file)
        N_progs = len(program_list)
    except Exception as e:
        print(f"Error loading program list: {e}")
        program_list = []
        N_progs = 0

    

    return

def send_summary_email():

    return

def daily_report():

    return 