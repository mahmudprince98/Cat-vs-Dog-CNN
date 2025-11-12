
import argparse, torch
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
import numpy as np

def load_checkpoint(path, device):
    ckpt = torch.load(path, map_location=device)
    model_name = ckpt.get("model_name","simple")
    img_size = ckpt.get("img_size",160)
    class_names = ckpt.get("class_names", ["cat","dog"])

    if model_name == "simple":
        # rebuild simple net
        class SimpleCNN(nn.Module):
            def __init__(self, num_classes=2):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                    nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                )
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(64*(20)*(20), 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, len(class_names))
                )
            def forward(self, x):
                x = self.features(x)
                return self.classifier(x)
        model = SimpleCNN(len(class_names))
    elif model_name == "resnet18":
        m = models.resnet18(weights=None)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, len(class_names))
        model = m
    else:
        raise ValueError("Unknown model in checkpoint")

    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval().to(device)

    tf = transforms.Compose([
        transforms.Resize((img_size,img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    return model, tf, class_names

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--image_path", required=True)
    ap.add_argument("--class_names", type=str, default=None, help='Optional override, e.g. "cat,dog"')
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tf, saved_names = load_checkpoint(args.weights, device)
    class_names = [s.strip() for s in args.class_names.split(",")] if args.class_names else saved_names

    img = Image.open(args.image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    logits = model(x)[0].cpu().numpy()
    probs = np.exp(logits - logits.max()); probs = probs / probs.sum()
    pred = int(probs.argmax())
    print(f"Prediction: {class_names[pred]}  (confidence={probs[pred]:.4f})")

if __name__ == "__main__":
    main()
