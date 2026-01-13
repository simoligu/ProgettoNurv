import torch
from torchvision import models, transforms as t
from PIL import Image
from typing import Optional, Tuple
import numpy as np

class SimpleClassifier:
    def __init__(self, device=None, num_classes=5):
        # 🔹 Scelta del device
        if device is None:
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")
            elif torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        print(f"[INFO] Classifier running on: {self.device}")

        # 🔹 Modello pre-addestrato ResNet18
        model = models.resnet18(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

        self.model = model.to(self.device)
        self.transform = t.Compose([
            t.Resize((224, 224)),
            t.ToTensor(),
            t.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def infer_image(self, pil_img: Image.Image) -> Tuple[int, float]:
        """
        Inferisce la classe dell'immagine PIL.
        Ritorna (predicted_class_index, confidence)
        """
        self.model.eval()

        # Converte in RGB se necessario
        if pil_img.mode != 'RGB':
            pil_img = pil_img.convert('RGB')

        x = self.transform(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(x)
            # Calcolo della probabilità (Softmax)
            probs = torch.nn.functional.softmax(logits, dim=1)

            # Estrazione della classe con la massima probabilità
            confidence, predicted_class_tensor = torch.max(probs, 1)

        return predicted_class_tensor.item(), confidence.item()