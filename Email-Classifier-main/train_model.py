import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
from preprocess import load_and_preprocess_data

def create_model(max_words, max_len, num_classes):
    model = Sequential([
        Embedding(max_words, 100, input_length=max_len),
        GlobalAveragePooling1D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model():
    # Load and preprocess data
    X_train, X_test, y_train, y_test, tokenizer = load_and_preprocess_data(
        'email_classification_dataset.csv',
        max_words=10000,
        max_len=100
    )
    
    # Create and train model
    model = create_model(max_words=10000, max_len=100, num_classes=3)
    
    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        epochs=10,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stopping]
    )
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"\nTest accuracy: {test_accuracy:.4f}")
    
    # Save the model
    model.save('email_classifier_model.h5')
    print("Model saved successfully!")
    
    return history

if __name__ == "__main__":
    history = train_model() 