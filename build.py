import os
import shutil
import subprocess
from pathlib import Path

def setup_environment():
    # Create necessary directories
    os.makedirs('functions', exist_ok=True)
    os.makedirs('frontend/build', exist_ok=True)
    
    # Copy frontend files to build directory
    frontend_src = Path('frontend')
    frontend_dest = Path('frontend/build')
    
    for file in frontend_src.glob('*'):
        if file.is_file():
            shutil.copy2(file, frontend_dest / file.name)
    
    # Create serverless function
    with open('functions/seo.py', 'w') as f:
        f.write('''
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from keybert import KeyBERT
import nltk
from openai import OpenAI
import language_tool_python

app = Flask(__name__)
CORS(app)

# Initialize components
nltk.download('punkt')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
kw_model = KeyBERT()
language_tool = language_tool_python.LanguageTool('en-US')

# Your API keys
API_KEY = os.environ.get('GOOGLE_API_KEY')
CX = os.environ.get('GOOGLE_CX')
OPENAI_KEY = os.environ.get('OPENAI_API_KEY')

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_KEY)

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        # Your existing analyze logic here
        return jsonify({"result": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        # Your existing generate logic here
        return jsonify({"result": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run()
''')

if __name__ == '__main__':
    setup_environment()
    print("Build process completed successfully!")
