import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class ColorizationDataset(Dataset):
    def __init__(self, root_dir, size=128):
        self.pairs = []
        self.size = size

        # Folder pairs (gray → color)
        folders = [
            ("gray_color_1", "color_1"),
            ("gray_black_1", "black_1"),
            ("gray_brown_1", "brown_1"),
            ("gray_white_1", "white_1")
        ]

        for gray_f, color_f in folders:
            gray_path = os.path.join(root_dir, gray_f)
            color_path = os.path.join(root_dir, color_f)

            if not os.path.exists(gray_path) or not os.path.exists(color_path):
                print(f"⚠️ Skipping {gray_f}/{color_f} (folder not found)")
                continue

            gray_files = sorted(os.listdir(gray_path))
            color_files = sorted(os.listdir(color_path))

            min_len = min(len(gray_files), len(color_files))

            for i in range(min_len):
                gray_file = os.path.join(gray_path, gray_files[i])
                color_file = os.path.join(color_path, color_files[i])

                self.pairs.append((gray_file, color_file))

        print(f"✅ Total pairs loaded: {len(self.pairs)}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        gray_path, color_path = self.pairs[idx]

        # Load images
        gray_img = Image.open(gray_path).convert("L")
        color_img = Image.open(color_path).convert("RGB")

        # Resize
        gray_img = gray_img.resize((self.size, self.size))
        color_img = color_img.resize((self.size, self.size))

        # Convert to tensor
        gray_tensor = TF.to_tensor(gray_img)   # (1, H, W)
        color_tensor = TF.to_tensor(color_img) # (3, H, W)

        # Normalize to [-1, 1] (VERY IMPORTANT for GAN 🔥)
        gray_tensor = (gray_tensor - 0.5) / 0.5
        color_tensor = (color_tensor - 0.5) / 0.5

        return gray_tensor, color_tensor