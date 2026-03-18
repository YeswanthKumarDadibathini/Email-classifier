import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import pickle

def load_and_preprocess_data(csv_file, max_words=10000, max_len=100):
    # Load the dataset
    df = pd.read_csv(csv_file)
    
    # Convert categories to numerical labels
    category_map = {'Work': 0, 'Personal': 1, 'Spam': 2}
    df['category'] = df['category'].map(category_map)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        df['email_text'], df['category'], test_size=0.2, random_state=42
    )
    
    # Initialize tokenizer
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(X_train)
    
    # Convert text to sequences
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    # Pad sequences
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_len)
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_len)
    
    # Save tokenizer for later use
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return X_train_pad, X_test_pad, y_train, y_test, tokenizer

def preprocess_text(text, tokenizer, max_len=100):
    # Convert text to sequence
    seq = tokenizer.texts_to_sequences([text])
    # Pad sequence
    padded = pad_sequences(seq, maxlen=max_len)
    return padded

if __name__ == "__main__":
    # Test the preprocessing
    X_train, X_test, y_train, y_test, tokenizer = load_and_preprocess_data('email_classification_dataset.csv')
    print("Data preprocessing completed successfully!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}") 