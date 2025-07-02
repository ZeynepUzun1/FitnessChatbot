from flask import Flask, request, jsonify, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')
app = Flask(__name__)

MAX_SEQUENCE_LENGTH = 100

with open('./tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

model = tf.keras.models.load_model('./sentiment_model.h5')

label_map = {0: "happy", 1: "sad"}

def predict_sentiment_from_text(text):
    sequences = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post', truncating='post')
    pred_prob = model.predict(padded)[0][0]
    pred_class = int(pred_prob > 0.5)
    sentiment = label_map[pred_class]
    return sentiment, float(pred_prob)

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']
    sentiment, confidence = predict_sentiment_from_text(text)
    return jsonify({
        'text': text,
        'sentiment': sentiment,
        'confidence': confidence
    })

from openai import OpenAI

client = OpenAI()

def get_openai_response(user_message, sentiment):
    prompt = f"""
You are a helpful fitness coach assistant. 
The user has the following sentiment: {sentiment}.
Their message is: "{user_message}"

Based on this, generate a supportive, personalized and short fitness schedule or motivational message to encourage the user.
Keep it positive and empathetic. Format the entire response using Markdown.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a friendly fitness coach."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message', '')
    if not user_message:
        return jsonify({'error': 'No message provided'}), 400

    sentiment, confidence = predict_sentiment_from_text(user_message)

    response_text = get_openai_response(user_message, sentiment)

    return jsonify({
        'response': response_text,
        'sentiment': sentiment,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=True)
