#!/usr/bin/env python
"""
Email Classifier - Command Line Prediction Tool

Usage:
    python predict.py "Your email text here"
    python predict.py --file sample_emails.txt

This script loads the trained model and makes predictions on input text.
"""

import sys
import os
import argparse
import tensorflow as tf
import numpy as np
import pickle
from preprocess import preprocess_text

def load_model_and_tokenizer():
    """Load the trained model and tokenizer."""
    try:
        if not os.path.exists('email_classifier_model.h5'):
            print("Error: Model file not found. Please run train_model.py first.")
            return None, None
            
        if not os.path.exists('tokenizer.pickle'):
            print("Error: Tokenizer file not found. Please run preprocess.py first.")
            return None, None
        
        model = tf.keras.models.load_model('email_classifier_model.h5')
        with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
            
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model or tokenizer: {str(e)}")
        return None, None

def predict_category(email_text, model, tokenizer):
    """Predict the category of an email text."""
    # Category mapping
    categories = ['Work', 'Personal', 'Spam']
    
    # Preprocess the text
    processed_text = preprocess_text(email_text, tokenizer)
    
    # Make prediction
    predictions = model.predict(processed_text, verbose=0)
    predicted_class = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class])
    
    # Get probabilities for all classes
    probabilities = {
        category: float(prob) 
        for category, prob in zip(categories, predictions[0])
    }
    
    return {
        'category': categories[predicted_class],
        'confidence': confidence,
        'probabilities': probabilities
    }

def main():
    parser = argparse.ArgumentParser(description='Predict email categories')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--text', type=str, help='Email text to classify')
    group.add_argument('--file', type=str, help='File containing email texts (one per line)')
    
    args = parser.parse_args()
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer()
    if model is None or tokenizer is None:
        sys.exit(1)
    
    if args.text:
        # Single prediction
        result = predict_category(args.text, model, tokenizer)
        
        print("\n====== Email Classification Result ======")
        print(f"Text: {args.text[:50]}..." if len(args.text) > 50 else f"Text: {args.text}")
        print(f"Category: {result['category']}")
        print(f"Confidence: {result['confidence'] * 100:.2f}%")
        print("\nProbabilities:")
        for category, prob in result['probabilities'].items():
            print(f"  - {category}: {prob * 100:.2f}%")
            
    elif args.file:
        # Batch prediction
        if not os.path.exists(args.file):
            print(f"Error: File {args.file} not found.")
            sys.exit(1)
            
        with open(args.file, 'r') as f:
            emails = [line.strip() for line in f if line.strip()]
        
        print(f"\nProcessing {len(emails)} emails...\n")
        results = []
        
        for i, email in enumerate(emails):
            result = predict_category(email, model, tokenizer)
            results.append(result)
            
            print(f"Email {i+1}: {email[:30]}..." if len(email) > 30 else f"Email {i+1}: {email}")
            print(f"  Category: {result['category']}")
            print(f"  Confidence: {result['confidence'] * 100:.2f}%\n")
        
        # Print summary
        categories = ['Work', 'Personal', 'Spam']
        counts = {category: sum(1 for r in results if r['category'] == category) for category in categories}
        
        print("\n====== Batch Classification Summary ======")
        for category, count in counts.items():
            print(f"{category}: {count} emails ({count/len(emails)*100:.1f}%)")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__)
        sys.exit(0)
    main() 