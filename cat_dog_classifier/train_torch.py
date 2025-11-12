
import os, argparse
import torch, torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm

def make_loaders(data_dir, img_size=160, batch_size=32, num_workers=2):
    tf_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    tf_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=tf_train)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=tf_eval)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, train_ds.classes

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
            nn.Linear(64*(20)*(20), 128),  # for img_size=160 -> after 3 pools: 160->80->40->20
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def build_model(model_name, num_classes):
    if model_name == "simple":
        return SimpleCNN(num_classes)
    elif model_name == "resnet18":
        m = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_f = m.fc.in_features
        m.fc = nn.Linear(in_f, num_classes)
        return m
    else:
        raise ValueError("model must be 'simple' or 'resnet18'")

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses, preds_all, labels_all = [], [], []
    for imgs, labels in tqdm(loader, desc="Train", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        preds_all.extend(logits.argmax(1).detach().cpu().tolist())
        labels_all.extend(labels.detach().cpu().tolist())
    return sum(losses)/len(losses), accuracy_score(labels_all, preds_all)

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    losses, preds_all, labels_all = [], [], []
    for imgs, labels in tqdm(loader, desc="Val", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)
        losses.append(loss.item())
        preds_all.extend(logits.argmax(1).detach().cpu().tolist())
        labels_all.extend(labels.detach().cpu().tolist())
    return sum(losses)/len(losses), accuracy_score(labels_all, preds_all)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=160)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--model", type=str, default="simple", choices=["simple","resnet18"])
    ap.add_argument("--out_dir", type=str, default="runs_torch")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    train_loader, val_loader, classes = make_loaders(args.data_dir, args.img_size, args.batch_size)
    model = build_model(args.model, num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc, best_path = 0.0, os.path.join(args.out_dir, "best.pt")

    for epoch in range(1, args.epochs+1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Train | loss {tr_loss:.4f} acc {tr_acc:.4f}")
        print(f"Val   | loss {val_loss:.4f} acc {val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "model_name": args.model,
                "img_size": args.img_size,
                "class_names": classes
            }, best_path)
            print(f"[INFO] Saved best to {best_path} (acc={best_acc:.4f})")

    print(f"[DONE] Best: {best_acc:.4f}  Weights: {best_path}")

if __name__ == "__main__":
    main()
