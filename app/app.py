import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import onnxruntime as ort
import json
from pathlib import Path

# --- Конфигурация ---
MODEL_PATH = Path("best_model.onnx")
CLASSES_PATH = Path("classes.json")
IMG_SIZE = 224

# Загрузка классов
with open(CLASSES_PATH, "r") as f:
    classes = json.load(f)

# Загрузка ONNX модели
session = ort.InferenceSession(str(MODEL_PATH))

# Трансформации (должны совпадать с валидационными из train.py)
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def predict_image(image):
    # Преобразование PIL Image в Tensor
    input_tensor = preprocess(image).unsqueeze(0) # Add batch dim
    # ONNX Runtime expects numpy array
    input_np = input_tensor.numpy()
    
    # Инференс
    inputs = {session.get_inputs()[0].name: input_np}
    outputs = session.run(None, inputs)
    logits = outputs[0][0]
    
    # Постобработка
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=0)
    confidences = {classes[i]: float(probs[i]) for i in range(len(classes))}
    
    return confidences

# --- Интерфейс Gradio ---
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=2),
    title="Классификатор: Кружка, Манга, Печенька, Игрушечный пистолет",
    description="Загрузите изображение для классификации. Модель работает на CPU через ONNX."
)

if __name__ == "__main__":
    interface.launch()
