Project Overview

The REAL or MORPH Image Detection System is a deep learning–based forensic application designed to classify facial images as REAL or MORPHED.
With the rapid growth of image manipulation, face morphing has become a serious threat to biometric security systems such as passport verification, border control, and digital identity authentication.
This project aims to detect such morphing attacks using Convolutional Neural Networks (CNNs) and computer vision techniques.

🎯 Objectives

Detect whether a facial image is genuine (REAL) or digitally morphed (MORPHED)

Learn subtle texture and identity-level inconsistencies using CNN

Provide a desktop-based GUI for easy image verification

Support common image formats like JPG, PNG, BMP

Perform inference locally without cloud dependency

🧠 Technologies Used

Programming Language: Python

Deep Learning Framework: PyTorch

Computer Vision: OpenCV, PIL

GUI: Tkinter

Libraries: NumPy, TorchVision

Development Tools: VS Code / Jupyter Notebook

🗂️ Dataset

Total Images: ~40,000

20,000 REAL face images

20,000 MORPHED face images

Sources include Kaggle datasets and self-generated morph images

Images were resized to 224×224, normalized, and cleaned

Dataset split:

Training: 60%

Validation: 15%

Testing: 15%

🏗️ System Architecture

Image Input – User uploads a facial image

Preprocessing – Resize, normalize, and format image

CNN Model – Extracts features and classifies image

Prediction Module – Determines REAL or MORPHED

GUI Output – Displays result with confidence score

⚙️ Methodology

Binary image classification using CNN

Supervised learning with Cross-Entropy Loss

Adam optimizer with learning rate scheduling

Data augmentation to improve generalization

Model trained to detect:

Texture inconsistencies

Blending artifacts

Identity-level mismatches

📊 Results

Achieved approximately 60% accuracy

Model successfully detects many morphing cases

Performance limited by:

High-quality GAN-based morphs

Subtle blending artifacts

Future improvements can enhance accuracy further

🖥️ Graphical User Interface (GUI)

Simple desktop interface built with Tkinter

Upload image via file dialog

Displays:

Uploaded image

Prediction result (REAL / MORPHED)

Confidence percentage

Color-coded output for better clarity

📁 Project Structure
├── dataset/
│   ├── real/
│   └── morphed/
├── model_pytorch.py
├── train.py
├── app.py
├── requirements.txt
└── README.md

🚀 How to Run


Install dependencies

pip install -r requirements.txt

Run GUI application

python app.py

📌 Limitations

Accuracy is moderate (~60%)

Struggles with advanced GAN-based morphs

Works only on static images

Not designed for video or real-time camera streams

🔮 Future Scope

Improve accuracy using deeper CNN or Vision Transformers

Integrate frequency-domain feature analysis

Support video-based morph detection

Mobile and web deployment

Real-time biometric system integration
