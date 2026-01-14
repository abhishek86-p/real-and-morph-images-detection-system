import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import torch
import torch.nn.functional as F
from torchvision import transforms, datasets
from model_pytorch import build_model
import os
import json

MODEL_PATH = "morph_detector.pt"
CLASSES_PATH = "classes.json"

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ------------------ Load Class Mapping ------------------
if os.path.exists(CLASSES_PATH):
    with open(CLASSES_PATH, "r") as f:
        classes = json.load(f)
else:
    train_dir = "dataset/train"

    if os.path.isdir(train_dir):
        td = datasets.ImageFolder(
            train_dir,
            transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor()
            ])
        )
        classes = td.classes
    else:
        classes = ["REAL", "MORPHED"]

idx_to_class = {i: c for i, c in enumerate(classes)}

# ------------------ Build & Load Model ------------------
num_classes = len(classes)
model = build_model(num_classes=num_classes)

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
else:
    print("Warning: model file not found. Predictions will be random until you train and save morph_detector.pt")

model.to(device)
model.eval()

# ------------------ Transform ------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------ Prediction Function ------------------
def predict(path):
    img = Image.open(path).convert("RGB")
    t = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(t)
        prob = F.softmax(out, dim=1)[0].cpu().numpy()

    idx = int(prob.argmax())
    label = idx_to_class[idx].upper()
    confidence = float(prob[idx])

    return label, confidence, prob


# ------------------ Tkinter UI ------------------
root = tk.Tk()
root.title("REAL vs MORPHED — Detector")
root.geometry("620x790")
root.resizable(False, False)
root.configure(bg="#111827")

# Top frame
top = tk.Frame(root, bg="#0b1220", padx=12, pady=12)
top.pack(fill="x")

title = tk.Label(
    top,
    text="REAL vs MORPHED Detector",
    font=("Segoe UI", 18, "bold"),
    fg="white",
    bg="#0b1220"
)
title.pack(side="left")

sub = tk.Label(
    top,
    text="(from-scratch CNN)",
    font=("Segoe UI", 9),
    fg="#cbd5e1",
    bg="#0b1220"
)
sub.pack(side="left", padx=10)

# Image display card
card = tk.Frame(root, bg="#0b1220", padx=10, pady=10)
card.pack(pady=18)

img_label = tk.Label(card, bg="#111827")
img_label.pack()

# Prediction panel
panel = tk.Frame(root, bg="#0b1220", padx=12, pady=12)
panel.pack(fill="x", pady=6)

result_label = tk.Label(
    panel,
    text="Select an image to predict",
    font=("Segoe UI", 14),
    fg="#e2e8f0",
    bg="#0b1220"
)
result_label.pack(pady=8)

prob_frame = tk.Frame(panel, bg="#0b1220")
prob_frame.pack(fill="x", pady=6)

prob_text = tk.Label(
    prob_frame,
    text="Confidence:",
    font=("Segoe UI", 11),
    fg="#e2e8f0",
    bg="#0b1220"
)
prob_text.pack(side="left")

progress = ttk.Progressbar(
    prob_frame, orient="horizontal",
    length=300,
    mode="determinate"
)
progress.pack(side="left", padx=10)

status_label = tk.Label(panel, text="", font=("Segoe UI", 12, "bold"), bg="#0b1220")
status_label.pack(pady=6)


# ------------------ Select Image Function ------------------
def choose_img():
    path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if not path:
        return

    img = Image.open(path).convert("RGB")
    display = img.copy()
    display.thumbnail((360, 360))

    tk_img = ImageTk.PhotoImage(display)
    img_label.config(image=tk_img)
    img_label.image = tk_img

    label, conf, probs = predict(path)

    pct = conf * 100
    progress['value'] = pct
    result_label.config(text=f"{label}  —  {pct:.2f}%")

    # Status color
    if label == "REAL":
        status_label.config(text="Genuine face detected", fg="#34d399")
    else:
        status_label.config(text="Possible morphed face", fg="#fb7185")

    # Show breakdown
    details = ""
    for i, p in enumerate(probs):
        cname = idx_to_class[i].upper()
        details += f"{cname}: {p*100:.2f}%   "

    detail_lbl.config(text=details)


# ------------------ Buttons ------------------
btn_frame = tk.Frame(root, bg="#111827")
btn_frame.pack(pady=12)

select_btn = tk.Button(
    btn_frame,
    text="Select Image",
    command=choose_img,
    font=("Segoe UI", 11),
    bg="#06b6d4",
    fg="black",
    padx=12,
    pady=6
)
select_btn.pack(side="left", padx=8)

quit_btn = tk.Button(
    btn_frame,
    text="Exit",
    command=root.destroy,
    font=("Segoe UI", 11),
    bg="#ef4444",
    fg="white",
    padx=12,
    pady=6
)
quit_btn.pack(side="left", padx=8)

detail_lbl = tk.Label(
    root,
    text="",
    font=("Segoe UI", 10),
    fg="#cbd5e1",
    bg="#111827"
)
detail_lbl.pack(pady=8)

footer = tk.Label(
    root,
    text="Train the model with `python train.py` then open this app.",
    font=("Segoe UI", 9),
    fg="#94a3b8",
    bg="#111827"
)
footer.pack(side="bottom", pady=10)

root.mainloop()
