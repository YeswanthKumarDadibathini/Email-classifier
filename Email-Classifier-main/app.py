from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import pickle
import os
from preprocess import preprocess_text

app = Flask(__name__)
CORS(app)

# Global variables for model and tokenizer
model = None
tokenizer = None
categories = ['Work', 'Personal', 'Spam']

def load_model_and_tokenizer():
    global model, tokenizer
    try:
        if os.path.exists('email_classifier_model.h5'):
            model = tf.keras.models.load_model('email_classifier_model.h5')
            print("Model loaded successfully!")
        else:
            print("Warning: Model file not found. Please run train_model.py first.")
            
        if os.path.exists('tokenizer.pickle'):
            with open('tokenizer.pickle', 'rb') as handle:
                tokenizer = pickle.load(handle)
            print("Tokenizer loaded successfully!")
        else:
            print("Warning: Tokenizer file not found. Please run preprocess.py first.")
            
        return model is not None and tokenizer is not None
    except Exception as e:
        print(f"Error loading model or tokenizer: {str(e)}")
        return False

# Try to load on startup
model_loaded = load_model_and_tokenizer()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/classifier')
def classifier():
    return render_template('classifier.html')

@app.route('/classify', methods=['POST'])
def classify_email():
    try:
        global model, tokenizer, model_loaded
        
        # If model or tokenizer is not loaded, try to load again
        if not model_loaded:
            model_loaded = load_model_and_tokenizer()
            if not model_loaded:
                return jsonify({
                    'error': 'Model or tokenizer not available. Please run preprocess.py and train_model.py first.'
                }), 503
        
        data = request.get_json()
        email_text = data.get('email_text', '')
        
        if not email_text:
            return jsonify({'error': 'No email text provided'}), 400
        
        # Preprocess the text
        processed_text = preprocess_text(email_text, tokenizer)
        
        # Make prediction
        predictions = model.predict(processed_text)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Get probabilities for all classes
        class_probabilities = {
            category: float(prob) 
            for category, prob in zip(categories, predictions[0])
        }
        
        return jsonify({
            'category': categories[predicted_class],
            'confidence': confidence,
            'probabilities': class_probabilities
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True) 