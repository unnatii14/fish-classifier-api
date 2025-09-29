<<<<<<< HEAD
# fish-classifier-api
=======
# 🐠 Fish Classifier API

AI-powered fish species classification using deep learning with EfficientNet-B0. This API can classify fish species and find similar fish in the database using image embeddings.

## 🌐 Live Demo

- **API**: [https://fish-api-md7q.onrender.com](https://fish-api-md7q.onrender.com)
- **API Documentation**: [https://fish-api-md7q.onrender.com/docs](https://fish-api-md7q.onrender.com/docs)
- **Health Check**: [https://fish-api-md7q.onrender.com/health](https://fish-api-md7q.onrender.com/health)

## ✨ Features

- ** Species Classification**: Classify fish species with confidence scores using EfficientNet-B0
- ** Similarity Search**: Find similar fish in the database using cosine similarity
- ** Web Interface**: User-friendly HTML interfaces for testing
- ** REST API**: FastAPI-based backend with automatic Swagger documentation
- ** Cloud Deployed**: Hosted on Render with automatic deployments

## 🏗️ Architecture

- **Backend**: FastAPI with Python 3.13
- **Model**: EfficientNet-B0 (pre-trained on ImageNet, fine-tuned for fish classification)
- **Image Processing**: PIL + torchvision transforms
- **Similarity Search**: Scikit-learn cosine similarity on image embeddings
- **Deployment**: Render cloud platform

## 🚀 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Welcome message |
| `GET` | `/health` | Health check status |
| `POST` | `/predict` | Classify fish species |
| `POST` | `/find-similar` | Find similar fish (top_k parameter) |

## 🖥️ Web Interface

### Simple Test Interface
- Clean, minimal interface for basic testing
- Upload image → Get predictions
- Real-time results with confidence scores

### Advanced Web Interface  
- Modern UI with drag & drop support
- Tabbed interface for predictions and similarity search
- Progress indicators and error handling
- Responsive design

## 🛠️ Local Development

### Prerequisites
- Python 3.9+
- pip or conda

### Setup
```bash
# Clone the repository
git clone https://github.com/unnatii14/fish-classifier-api.git
cd fish-classifier-api

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

### Local Testing
1. Start the API server (port 8000)
2. Open `simple_test_interface.html` in your browser
3. For development, update API_BASE to `http://127.0.0.1:8000`

## 🔧 Model Details

- **Architecture**: EfficientNet-B0
- **Input Size**: 224×224 RGB images
- **Preprocessing**: ImageNet normalization
- **Classes**: Multiple fish species (see label_mapping.json)
- **Similarity**: Feature embeddings + cosine similarity

## 🔄 Deployment

This project is automatically deployed on [Render](https://render.com) with:
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Auto-deploy**: Enabled on main branch commits


## 🙏 Acknowledgments

- EfficientNet architecture by Google Research
- FastAPI framework for the REST API
- Render platform for cloud hosting
- PyTorch and torchvision for deep learning infrastructure

---

### 🔗 Quick Links

- ** Live API**: https://fish-api-md7q.onrender.com
- ** API Docs**: https://fish-api-md7q.onrender.com/docs
- ** Repository**: https://github.com/unnatii14/fish-classifier-api
- ** Contact**: Open an issue for questions or support

*Happy Fish Classifying! 🐠🚀*
>>>>>>> 96b285990e1d1b8cbb289d94f5f451d968979122
