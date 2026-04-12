import cv2
import torch
from PIL import Image
import torchvision.transforms as transforms
from model import UNetGenerator

def colorize_frame(frame, model, transform, device):
    # frame: BGR numpy array (H,W,3) from OpenCV
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_pil = Image.fromarray(gray)
    gray_tensor = transform(gray_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        fake_ab = model(gray_tensor)
    L = gray_tensor * 100.0
    # convert L + ab to RGB (reuse lab_to_rgb from above)
    rgb = lab_to_rgb(L.cpu(), fake_ab.cpu())
    return cv2.cvtColor(np.array(rgb), cv2.COLOR_RGB2BGR)

def colorize_video(input_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetGenerator().to(device)
    model.load_state_dict(torch.load("pix2pix_generator.pth", map_location=device))
    model.eval()
    transform = transforms.Compose([transforms.Resize((128,128)), transforms.ToTensor()])

    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(3)), int(cap.get(4))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        color_frame = colorize_frame(frame, model, transform, device)
        # resize back to original dimensions
        color_frame = cv2.resize(color_frame, (width, height))
        out.write(color_frame)
    cap.release()
    out.release()

# Example
colorize_video("input.mp4", "output_colorized.mp4")