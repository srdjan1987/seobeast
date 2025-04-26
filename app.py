from flask import Flask, render_template, request, jsonify
import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from keybert import KeyBERT
import nltk
from openai import OpenAI
import json
from urllib.parse import urlparse
import language_tool_python

nltk.download('punkt')
nltk.download('words')
nltk.download('averaged_perceptron_tagger')
kw_model = KeyBERT()

# Initialize OpenAI client
client = OpenAI(api_key="YOUR_OPENAI_API_KEY")

# Initialize language tool for grammar checking
language_tool = language_tool_python.LanguageTool('en-US')

app = Flask(__name__, static_folder='frontend', template_folder='frontend')

# Move all the API methods from seo.py to this file
# ... (copy all the methods from seo.py here)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze():
    data = request.json
    result = api.analyse(data)
    return jsonify(json.loads(result))

@app.route('/api/score', methods=['POST'])
def score():
    data = request.json
    result = api.score(data['html'], data['tfidf'], data['lsi'], data['keywords'])
    return jsonify(result)

@app.route('/api/generate_content', methods=['POST'])
def generate_content():
    data = request.json
    result = api.generate_content(data)
    return jsonify({'content': result})

@app.route('/api/generate_headlines', methods=['POST'])
def generate_headlines():
    data = request.json
    result = api.generate_headline_suggestions(data['keyword'])
    return jsonify(result)

@app.route('/api/build_topic_clusters', methods=['POST'])
def build_topic_clusters():
    data = request.json
    result = api.build_topic_clusters(data['keyword'])
    return jsonify(result)

@app.route('/api/validate_outline', methods=['POST'])
def validate_outline():
    data = request.json
    result = api.validate_outline(data['headings'])
    return jsonify(result)

@app.route('/api/detect_fluff', methods=['POST'])
def detect_fluff():
    data = request.json
    result = api.detect_fluff(data['content'])
    return jsonify(result)

@app.route('/api/compression_score', methods=['POST'])
def compression_score():
    data = request.json
    result = api.compression_score(data['content'])
    return jsonify(result)

@app.route('/api/expand_lsi', methods=['POST'])
def expand_lsi():
    data = request.json
    result = api.expand_lsi_keywords(data['content'])
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True) 