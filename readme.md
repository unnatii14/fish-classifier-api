# 🐠 Fish Classifier API

AI-powered fish species classification using deep learning with EfficientNet.

## Features

- **Species Prediction**: Classify fish species with confidence scores
- **Similarity Search**: Find similar fish in the database
- **Web Interface**: User-friendly HTML interfaces for testing
- **REST API**: FastAPI-based backend with automatic documentation

## Live Demo

🌐 **API**: [https://your-app.onrender.com](https://your-app.onrender.com)
📚 **API Docs**: [https://your-app.onrender.com/docs](https://your-app.onrender.com/docs)

## Local Development

### Prerequisites
- Python 3.9+
- pip

### Setup
```bash
# Clone the repository
git clone https://github.com/yourusername/fish-classifier-api.git
cd fish-classifier-api

# Install dependencies
cd fish-classifier-backend
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Testing
Open `simple_test_interface.html` or `fish_classifier_web_interface.html` in your browser.

## API Endpoints

- `GET /` - Welcome message
- `GET /health` - Health check
- `POST /predict` - Classify fish species
- `POST /find-similar` - Find similar fish

## Model Information

- **Architecture**: EfficientNet-B0
- **Input Size**: 224x224 RGB images
- **Preprocessing**: ImageNet normalization

## Deployment

This app is deployed on [Render](https://render.com) with automatic builds from the main branch.
