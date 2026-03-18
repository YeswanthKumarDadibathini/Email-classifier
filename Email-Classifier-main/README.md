# Email Classifier Using Neural Network

A full-stack web application that uses a neural network to classify emails into categories (Work, Personal, or Spam).

## Features

- Modern, responsive web interface
- Email classification using neural networks
- Real-time classification with confidence scores
- Visualized results using Chart.js

## Technologies Used

- **Frontend**: HTML, CSS, JavaScript, Bootstrap 5, Chart.js
- **Backend**: Python, Flask
- **Machine Learning**: TensorFlow/Keras
- **Data Processing**: NumPy, Pandas

## Installation

1. Clone this repository:
```
git clone https://github.com/chiru5190/Email-Classifier.git
cd email-classifier
```

2. Install required dependencies:
```
pip install -r requirements.txt
```

3. Run the preprocessing and model training (only needed once):
```
python preprocess.py
python train_model.py
```

## Usage

1. Start the Flask application:
```
python app.py
```

2. Open your browser and navigate to:
```
http://127.0.0.1:5000/
```

3. Use the application:
   - Navigate to the classifier page
   - Enter email text in the input field
   - Click "Classify Email"
   - View the predicted category and confidence scores

## Project Structure

```
├── app.py                     # Flask application
├── preprocess.py              # Data preprocessing logic
├── train_model.py             # Model training script
├── email_classifier_model.h5  # Trained neural network model (generated)
├── tokenizer.pickle           # Saved tokenizer (generated)
├── requirements.txt           # Python dependencies
├── templates/                 # HTML templates
│   ├── index.html             # Homepage
│   └── classifier.html        # Email classifier page
└── static/                    # Static files (CSS, JS, images)
```

## Model Architecture

The email classification model uses a neural network with the following architecture:
- Embedding layer
- Global Average Pooling
- Dense layers with dropout for regularization
- Softmax output layer (3 classes: Work, Personal, Spam)

## License

MIT

## Author


chiru5190
