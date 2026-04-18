# 🎨 Image & Video Colorization using Pix2Pix GAN

## 📌 Overview
This project performs automatic colorization of grayscale images and videos using a Pix2Pix GAN (Generative Adversarial Network).  
It also provides a Streamlit-based UI for real-time interaction and visualization.

---

## 🖼️ Sample Output
<img width="1280" height="526" alt="image" src="https://github.com/user-attachments/assets/7d856fb4-0238-4a7f-ad13-a394938c2b33" />

---

## 🚀 Features

- 🖼️ Image Colorization (Gray → Color)
- 🎥 Video Colorization (Frame-by-frame)
- 🔄 Color → Gray → Predict Mode
- ⚡ Real-time Streamlit UI
- 🎛️ Adaptive & Manual Enhancement
- 📊 Side-by-side Comparison
- 📥 Download Output

---

## 🧠 Model Architecture

- Generator: **U-Net**
- Discriminator: **PatchGAN**
- Loss Functions:
  - Adversarial Loss (GAN)
  - L1 Loss (Pixel-level accuracy)

---

## 📊 Training Details

- Dataset Size: **8600+ images**
- Epochs: **40**
- Framework: **PyTorch**
- Device: GPU (MPS / CUDA)

---

## 🛠️ Tech Stack

- Python
- PyTorch
- OpenCV
- Streamlit
- NumPy
- PIL

---

## ▶️ How to Run

### 1️⃣ Install Dependencies
```bash
pip install torch torchvision opencv-python streamlit numpy pillow
