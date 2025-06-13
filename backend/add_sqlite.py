import sqlite3
import json
import os

def add_data_to_db():
    db_path = os.path.join(os.path.dirname(__file__), 'db.sqlite3')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create and populate semantic_clustering_sentiment table
    semantic_json_path = os.path.join(os.path.dirname(__file__), 'Final_semantic_clustering_sentiment.json')
    if os.path.exists(semantic_json_path):
        print("Processing Final_semantic_clustering_sentiment.json...")
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS semantic_clustering_sentiment (
            id TEXT PRIMARY KEY,
            cleaned_text TEXT,
            createdAt TEXT,
            cluster_id INTEGER,
            sentiment TEXT,
            x REAL,
            y REAL
        )
        ''')

        with open(semantic_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                cursor.execute('''
                INSERT OR REPLACE INTO semantic_clustering_sentiment (id, cleaned_text, createdAt, cluster_id, sentiment, x, y)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (item['id'], item['cleaned_text'], item['createdAt'], item['cluster_id'], item['sentiment'], item['x'], item['y']))
        print("Finished processing Final_semantic_clustering_sentiment.json.")
    else:
        print(f"File not found: {semantic_json_path}")

    # Create and populate term_ngram_frequency table
    term_json_path = os.path.join(os.path.dirname(__file__), 'Final_term_ngram_frequency.json')
    if os.path.exists(term_json_path):
        print("Processing Final_term_ngram_frequency.json...")
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS term_ngram_frequency (
            date TEXT,
            term TEXT,
            frequency INTEGER,
            PRIMARY KEY (date, term)
        )
        ''')
        
        with open(term_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for date, terms in data.items():
                for term, frequency in terms.items():
                    cursor.execute('''
                    INSERT OR REPLACE INTO term_ngram_frequency (date, term, frequency)
                    VALUES (?, ?, ?)
                    ''', (date, term, frequency))
        print("Finished processing Final_term_ngram_frequency.json.")
    else:
        print(f"File not found: {term_json_path}")

    conn.commit()
    conn.close()
    print("Database has been updated successfully.")

if __name__ == '__main__':
    add_data_to_db()
