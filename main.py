from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import json
from sklearn.metrics.pairwise import cosine_similarity
import io
import os

app = FastAPI(title="Fish Classifier API", version="1.0.0")

# CORS for local testing, change for production!
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:8000", 
        "https://your-render-app.onrender.com",  # Replace with your actual Render URL
        "*"  # Remove this in production for security
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (224, 224)

model = None
feat_model = None
id_to_label = None
embeddings = None
emb_labels = None
emb_paths = None

transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def load_models():
    global model, feat_model, id_to_label, embeddings, emb_labels, emb_paths
    try:
        with open("label_mapping.json", "r") as f:
            label_mapping = json.load(f)
        id_to_label = {int(v): k for k, v in label_mapping.items()}
        num_classes = len(label_mapping)

        model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        model.load_state_dict(torch.load("best_model_efficientnet.pth", map_location=DEVICE), strict=False)
        model.eval().to(DEVICE)

        feat_model = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        feat_model.classifier[1] = nn.Linear(feat_model.classifier[1].in_features, num_classes)
        feat_model.load_state_dict(torch.load("best_model_efficientnet.pth", map_location=DEVICE), strict=False)
        feat_model.classifier = nn.Identity()
        feat_model.eval().to(DEVICE)

        embeddings = np.load("val_embeddings.npy")
        emb_labels = np.load("val_labels.npy")
        emb_paths = np.load("val_image_paths.npy")
    except Exception as e:
        print(f"Model loading error: {e}")
        raise

@app.on_event("startup")
def startup_event():
    load_models()

@app.get("/")
def root():
    return {"message": "Fish Classifier API is running on Render!"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        with torch.no_grad():
            t = transform(image).unsqueeze(0).to(DEVICE)
            out = model(t)
            probs = F.softmax(out, dim=1).cpu().numpy()[0]
            ids = probs.argsort()[::-1][:5]
        predictions = [{"species": id_to_label[int(i)], "confidence": float(probs[i])} for i in ids]
        return {"predictions": predictions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/find-similar")
async def find_similar(file: UploadFile = File(...), top_k: int = 5):
    try:
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        with torch.no_grad():
            t = transform(image).unsqueeze(0).to(DEVICE)
            q_feat = feat_model(t).cpu().numpy()
            if q_feat.ndim > 2:
                q_feat = q_feat.reshape(q_feat.shape[0], -1)
        emb = embeddings
        if emb.ndim > 2:
            emb = emb.reshape(emb.shape[0], -1)
        sims = cosine_similarity(q_feat, emb)[0]
        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_indices:
            similar_label_id = int(emb_labels[idx])
            results.append({
                "species": id_to_label.get(similar_label_id, f"ID {similar_label_id}"),
                "similarity": float(sims[idx]),
                "image_path": str(emb_paths[idx])
            })
        return {"similar_images": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Similarity search error: {str(e)}")