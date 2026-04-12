import io
import os
import tempfile
import time

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image


# -------------------- Model --------------------
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super().__init__()
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)

        self.dec4 = self.upconv_block(512, 256)
        self.dec3 = self.upconv_block(512, 128)
        self.dec2 = self.upconv_block(256, 64)
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh()
        )

    def conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def upconv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        d4 = self.dec4(e4)
        d4 = torch.cat([d4, e3], dim=1)

        d3 = self.dec3(d4)
        d3 = torch.cat([d3, e2], dim=1)

        d2 = self.dec2(d3)
        d2 = torch.cat([d2, e1], dim=1)

        d1 = self.dec1(d2)
        return d1


# -------------------- Device --------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -------------------- Load Model --------------------
@st.cache_resource
def load_model():
    device = get_device()
    model = UNetGenerator(in_channels=1, out_channels=3).to(device)

    checkpoint_paths = [
        "pix2pix_generator.pth",
        "pix2pix_generator_final.pth",
        "checkpoint_latest.pth",
        "checkpoint_epoch_20.pth",
        "checkpoint_epoch_15.pth",
        "checkpoint_epoch_10.pth",
        "checkpoint_epoch_5.pth",
    ]

    loaded_path = None
    for path in checkpoint_paths:
        if not os.path.exists(path):
            continue

        data = torch.load(path, map_location=device)

        if isinstance(data, dict) and "generator_state_dict" in data:
            model.load_state_dict(data["generator_state_dict"], strict=True)
        else:
            model.load_state_dict(data, strict=True)

        loaded_path = path
        break

    if loaded_path is None:
        st.error("No model file found. Put checkpoint or generator weights in this folder.")
        st.stop()

    model.eval()
    st.success(f"Loaded model: {loaded_path}")
    return model, device


# -------------------- Enhancement Helpers --------------------
def auto_enhance_image(pil_img):
    img_np = np.array(pil_img).astype(np.float32)
    gray = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    if laplacian_var < 40:
        sharp_strength = 1.3
    elif laplacian_var < 120:
        sharp_strength = 0.9
    else:
        sharp_strength = 0.4

    mean_val = np.mean(img_np)
    std_val = np.std(img_np)

    if std_val < 35:
        contrast_alpha = 1.18
    elif std_val < 55:
        contrast_alpha = 1.10
    else:
        contrast_alpha = 1.04

    if mean_val < 95:
        beta = 12
    elif mean_val > 190:
        beta = -8
    else:
        beta = 0

    kernel = np.array([
        [0, -0.5 * sharp_strength, 0],
        [-0.5 * sharp_strength, 3 * sharp_strength, -0.5 * sharp_strength],
        [0, -0.5 * sharp_strength, 0]
    ])

    img_np = cv2.filter2D(img_np, -1, kernel)
    img_np = np.clip(img_np, 0, 255)
    img_np = cv2.convertScaleAbs(img_np, alpha=contrast_alpha, beta=beta)
    return Image.fromarray(img_np)


def manual_enhance_image(pil_img, sharp_strength, contrast_alpha, contrast_beta):
    img_np = np.array(pil_img).astype(np.float32)

    kernel = np.array([
        [0, -0.5 * sharp_strength, 0],
        [-0.5 * sharp_strength, 3 * sharp_strength, -0.5 * sharp_strength],
        [0, -0.5 * sharp_strength, 0]
    ])

    img_np = cv2.filter2D(img_np, -1, kernel)
    img_np = np.clip(img_np, 0, 255)
    img_np = cv2.convertScaleAbs(img_np, alpha=contrast_alpha, beta=contrast_beta)
    return Image.fromarray(img_np)


def boost_video_colors(rgb_img, saturation_scale=1.45, value_scale=1.08):
    hsv = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] *= saturation_scale
    hsv[:, :, 2] *= value_scale
    hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2], 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


# -------------------- Video Frame Analysis --------------------
def analyze_video_frame_quality(gray_frame):
    mean_brightness = float(np.mean(gray_frame))
    contrast_std = float(np.std(gray_frame))
    laplacian_var = float(cv2.Laplacian(gray_frame, cv2.CV_64F).var())

    overexposed = mean_brightness > 225
    underexposed = mean_brightness < 30
    very_low_contrast = contrast_std < 18
    blurry = laplacian_var < 18

    severe_exposure = overexposed or underexposed
    low_quality = severe_exposure or (very_low_contrast and blurry)

    return {
        "mean_brightness": mean_brightness,
        "contrast_std": contrast_std,
        "laplacian_var": laplacian_var,
        "overexposed": overexposed,
        "underexposed": underexposed,
        "low_contrast": very_low_contrast,
        "blurry": blurry,
        "low_quality": low_quality,
    }


def preprocess_gray_frame(gray_frame, quality_info):
    gray = gray_frame.copy()

    if quality_info["low_contrast"] or quality_info["overexposed"] or quality_info["underexposed"]:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    if quality_info["blurry"]:
        blur1 = cv2.GaussianBlur(gray, (0, 0), 1.0)
        blur2 = cv2.GaussianBlur(gray, (0, 0), 2.0)
        gray = cv2.addWeighted(blur1, 1.4, blur2, -0.4, 0)

    gray = cv2.fastNlMeansDenoising(gray, None, 5, 7, 21)
    return gray


def soften_unreliable_video_prediction(rgb_pred, gray_frame, quality_info):
    gray_rgb = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
    gray_rgb_small = cv2.resize(
        gray_rgb,
        (rgb_pred.shape[1], rgb_pred.shape[0]),
        interpolation=cv2.INTER_CUBIC
    )

    if quality_info["overexposed"]:
        alpha = 0.65
    elif quality_info["underexposed"]:
        alpha = 0.68
    elif quality_info["blurry"] or quality_info["low_contrast"]:
        alpha = 0.72
    else:
        alpha = 0.75

    blended = cv2.addWeighted(rgb_pred, alpha, gray_rgb_small, 1 - alpha, 0)
    blended = cv2.convertScaleAbs(blended, alpha=1.08, beta=6)
    return boost_video_colors(blended, saturation_scale=1.15, value_scale=1.03)


# -------------------- Inference --------------------
def colorize_gray_pil(gray_img, model, device):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    gray_tensor = transform(gray_img)
    gray_tensor = (gray_tensor - 0.5) / 0.5
    gray_tensor = gray_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(gray_tensor)

    output = output.squeeze(0).cpu()
    output = (output + 1) / 2
    output_np = output.permute(1, 2, 0).numpy()
    output_np = np.clip(output_np, 0, 1)
    return Image.fromarray((output_np * 255).astype(np.uint8))


def apply_output_enhancement(rgb, enhance_mode, manual_params):
    rgb = cv2.convertScaleAbs(rgb, alpha=1.18, beta=10)
    rgb = boost_video_colors(rgb, saturation_scale=1.45, value_scale=1.08)

    if enhance_mode == "Auto (adaptive)":
        rgb = np.array(auto_enhance_image(Image.fromarray(rgb)))
        rgb = boost_video_colors(rgb, saturation_scale=1.10, value_scale=1.02)
    elif enhance_mode == "Manual (advanced)" and manual_params:
        rgb = np.array(
            manual_enhance_image(
                Image.fromarray(rgb),
                max(1.2, manual_params["sharp_strength"]),
                max(1.1, manual_params["contrast_alpha"]),
                manual_params["contrast_beta"] + 5,
            )
        )
        rgb = boost_video_colors(rgb, saturation_scale=1.10, value_scale=1.02)

    return rgb


# -------------------- Video Writer --------------------
def create_video_writer(output_path, fps, width, height):
    codecs = ["avc1", "mp4v"]
    for codec in codecs:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if writer.isOpened():
            return writer, codec
    return None, None


# -------------------- Video Processing --------------------
def process_video(input_path, gray_output_path, pred_output_path, model, device, enhance_mode, manual_params=None, mode="Gray -> Predict"):
    cap = cv2.VideoCapture(input_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 24
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    gray_writer, gray_codec = create_video_writer(gray_output_path, fps, width, height)
    pred_writer, pred_codec = create_video_writer(pred_output_path, fps, width, height)

    if gray_writer is None or pred_writer is None:
        cap.release()
        raise RuntimeError("Could not create video writer for output preview.")

    progress_bar = st.progress(0)
    status_text = st.empty()

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if mode == "Color -> Gray -> Predict":
            base_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            # If uploaded video is already grayscale/BW, this keeps the same luminance path.
            base_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        quality_info = analyze_video_frame_quality(base_gray)
        gray_prepared = preprocess_gray_frame(base_gray, quality_info)

        gray_bgr = cv2.cvtColor(gray_prepared, cv2.COLOR_GRAY2BGR)
        gray_bgr = cv2.resize(gray_bgr, (width, height), interpolation=cv2.INTER_CUBIC)
        gray_writer.write(gray_bgr)

        gray_pil = Image.fromarray(gray_prepared)
        rgb_pil = colorize_gray_pil(gray_pil, model, device)
        rgb = np.array(rgb_pil)

        if quality_info["low_quality"]:
            rgb = soften_unreliable_video_prediction(rgb, gray_prepared, quality_info)

        rgb = apply_output_enhancement(rgb, enhance_mode, manual_params)

        pred_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        pred_bgr = cv2.resize(pred_bgr, (width, height), interpolation=cv2.INTER_CUBIC)
        pred_writer.write(pred_bgr)

        frame_count += 1
        if total_frames > 0:
            progress_bar.progress(min(frame_count / total_frames, 1.0))

        if quality_info["overexposed"]:
            msg = "too much lighting"
        elif quality_info["underexposed"]:
            msg = "frame too dark"
        elif quality_info["blurry"]:
            msg = "object not clear"
        elif quality_info["low_contrast"]:
            msg = "low contrast"
        else:
            msg = "normal"

        status_text.text(f"Processing frame {frame_count}/{total_frames} - {msg}")

    cap.release()
    gray_writer.release()
    pred_writer.release()
    cv2.destroyAllWindows()
    time.sleep(0.3)

    return gray_codec, pred_codec


# -------------------- UI --------------------
st.set_page_config(page_title="Advanced Colorization", layout="wide")
st.title("Advanced Image and Video Colorization")

model, device = load_model()

enhance_mode = st.radio(
    "Enhancement Mode",
    ["None", "Auto (adaptive)", "Manual (advanced)"],
    index=1
)

manual_params = None
if enhance_mode == "Manual (advanced)":
    st.sidebar.header("Manual Enhancement Controls")
    sharp_strength = st.sidebar.slider("Sharpening strength", 0.0, 2.0, 1.0, 0.05)
    contrast_alpha = st.sidebar.slider("Contrast (alpha)", 0.5, 2.0, 1.05, 0.01)
    contrast_beta = st.sidebar.slider("Brightness (beta)", -50, 50, 5)
    manual_params = {
        "sharp_strength": sharp_strength,
        "contrast_alpha": contrast_alpha,
        "contrast_beta": contrast_beta,
    }

main_option = st.radio("Select Type", ["Image", "Video"])


# -------------------- IMAGE --------------------
if main_option == "Image":
    image_mode = st.radio("Select Mode", ["Color -> Gray -> Predict", "Gray -> Predict"])
    uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded:
        img = Image.open(uploaded).convert("RGB")

        if image_mode == "Color -> Gray -> Predict":
            original = img
            gray = img.convert("L")
        else:
            original = None
            gray = img.convert("L")

        result = colorize_gray_pil(gray, model, device)
        result_np = np.array(result)
        result_np = apply_output_enhancement(result_np, enhance_mode, manual_params)
        result = Image.fromarray(result_np)

        if image_mode == "Color -> Gray -> Predict":
            col1, col2, col3 = st.columns(3)
            col1.image(original, caption="Original", use_container_width=True)
            col2.image(gray, caption="Grayscale", use_container_width=True)
            col3.image(result, caption="Predicted", use_container_width=True)
        else:
            col1, col2 = st.columns(2)
            col1.image(gray, caption="Input Grayscale", use_container_width=True)
            col2.image(result, caption="Predicted", use_container_width=True)

        buf = io.BytesIO()
        result.save(buf, format="PNG")
        st.download_button("Download Colorized Image", buf.getvalue(), "colorized.png")


# -------------------- VIDEO --------------------
elif main_option == "Video":
    video_mode = st.radio("Select Mode", ["Color -> Gray -> Predict", "Gray -> Predict"])
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

    if uploaded_video:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_input:
            tmp_input.write(uploaded_video.read())
            input_path = tmp_input.name

        gray_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        pred_output_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

        with open(input_path, "rb") as f:
            input_video_bytes = f.read()

        st.subheader("Video Preview")

        if video_mode == "Color -> Gray -> Predict":
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown("### Original Video")
                st.video(input_video_bytes)
            with col2:
                st.markdown("### Grayscale Video")
                st.info("Grayscale video will appear here after processing.")
            with col3:
                st.markdown("### Predicted Video")
                st.info("Predicted video will appear here after processing.")
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Input Grayscale Video")
                st.video(input_video_bytes)
            with col2:
                st.markdown("### Predicted Video")
                st.info("Predicted video will appear here after processing.")

        if st.button("Start Colorization"):
            with st.spinner("Processing video... please wait"):
                process_video(
                    input_path,
                    gray_output_path,
                    pred_output_path,
                    model,
                    device,
                    enhance_mode,
                    manual_params,
                    mode=video_mode,
                )

            st.success("Video colorization complete")

            gray_video_bytes = None
            pred_video_bytes = None

            if os.path.exists(gray_output_path) and os.path.getsize(gray_output_path) > 0:
                with open(gray_output_path, "rb") as f:
                    gray_video_bytes = f.read()

            if os.path.exists(pred_output_path) and os.path.getsize(pred_output_path) > 0:
                with open(pred_output_path, "rb") as f:
                    pred_video_bytes = f.read()

            st.subheader("Prediction Result")

            if video_mode == "Color -> Gray -> Predict":
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.markdown("### Original Video")
                    st.video(input_video_bytes)

                with col2:
                    st.markdown("### Grayscale Video")
                    if gray_video_bytes:
                        st.video(gray_video_bytes)
                    else:
                        st.error("Grayscale preview video was not created correctly.")

                with col3:
                    st.markdown("### Predicted Video")
                    if pred_video_bytes:
                        st.video(pred_video_bytes)
                    else:
                        st.error("Predicted output video was not created correctly.")
            else:
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### Input Grayscale Video")
                    if gray_video_bytes:
                        st.video(gray_video_bytes)
                    else:
                        st.video(input_video_bytes)

                with col2:
                    st.markdown("### Predicted Video")
                    if pred_video_bytes:
                        st.video(pred_video_bytes)
                    else:
                        st.error("Predicted output video was not created correctly.")

            if pred_video_bytes:
                st.download_button(
                    "Download Colorized Video",
                    data=pred_video_bytes,
                    file_name="colorized_video.mp4",
                    mime="video/mp4"
                )

            try:
                os.unlink(input_path)
            except OSError:
                pass

            try:
                os.unlink(gray_output_path)
            except OSError:
                pass

            try:
                os.unlink(pred_output_path)
            except OSError:
                pass
