import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from model_pytorch import build_model

def train_model(train_dir, val_dir, epochs=8, batch_size=16, save_path="morph_detector.pt"):
    if torch.cuda.is_available():
        device = "cuda"
        print("GPU Found:", torch.cuda.get_device_name(0))
    else:
        device = "cpu"
    print("⚠ No GPU found, using CPU")

    print("Training on:", device)

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip()
    ])

    train_data = datasets.ImageFolder(train_dir, transform)
    val_data = datasets.ImageFolder(val_dir, transform)

    print("Class mapping:", train_data.class_to_idx)  # IMPORTANT

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0008)

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            loss = criterion(preds, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), save_path)
    print("✅ Model Saved as:", save_path)

if __name__ == "__main__":
    train_model("dataset/train", "dataset/val")
