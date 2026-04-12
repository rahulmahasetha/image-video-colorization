import time
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF

# -------------------- Dataset (grayscale → RGB) --------------------
class ColorizationRGBDataset(Dataset):
    def __init__(self, root_dir, size=128):
        self.pairs = []
        self.size = size
        folders = [("gray_color_1","color_1"), ("gray_black_1","black_1"),
                   ("gray_brown_1","brown_1"), ("gray_white_1","white_1")]
        for gray_f, color_f in folders:
            gray_path = os.path.join(root_dir, gray_f)
            color_path = os.path.join(root_dir, color_f)
            if not os.path.exists(gray_path) or not os.path.exists(color_path):
                continue
            # Pair by sorted index (fixes filename mismatch)
            gray_files = sorted(os.listdir(gray_path))
            color_files = sorted(os.listdir(color_path))
            min_len = min(len(gray_files), len(color_files))
            for i in range(min_len):
                self.pairs.append((
                    os.path.join(gray_path, gray_files[i]),
                    os.path.join(color_path, color_files[i])
                ))
        print(f"Found {len(self.pairs)} image pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        gray_img = Image.open(self.pairs[idx][0]).convert("L")
        color_img = Image.open(self.pairs[idx][1]).convert("RGB")
        gray_img = gray_img.resize((self.size, self.size), Image.BICUBIC)
        color_img = color_img.resize((self.size, self.size), Image.BICUBIC)
        gray_tensor = TF.to_tensor(gray_img)   # (1,H,W) in [0,1]
        color_tensor = TF.to_tensor(color_img) # (3,H,W) in [0,1]
        # Normalize to [-1, 1] for GAN stability
        gray_tensor = (gray_tensor - 0.5) / 0.5
        color_tensor = (color_tensor - 0.5) / 0.5
        return gray_tensor, color_tensor

# -------------------- Generator: U-Net (output 3 channels, Tanh) --------------------
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
            nn.LeakyReLU(0.2)
        )
    def upconv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
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

# -------------------- Discriminator: PatchGAN --------------------
class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 1, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)

# -------------------- Checkpoint helpers --------------------
def save_checkpoint(epoch, gen, disc, optimizer_G, optimizer_D, loss, filename):
    """Save full training state (model, optimizer, epoch)"""
    torch.save({
        'epoch': epoch,
        'generator_state_dict': gen.state_dict(),
        'discriminator_state_dict': disc.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_state_dict': optimizer_D.state_dict(),
        'loss': loss,
    }, filename)
    print(f"💾 Checkpoint saved: {filename}")

def load_checkpoint(filename, gen, disc, optimizer_G, optimizer_D, device):
    """Load training state and return starting epoch (0 if no checkpoint)"""
    if os.path.exists(filename):
        checkpoint = torch.load(filename, map_location=device)
        gen.load_state_dict(checkpoint['generator_state_dict'])
        disc.load_state_dict(checkpoint['discriminator_state_dict'])
        optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        optimizer_D.load_state_dict(checkpoint['optimizer_D_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f"🔄 Resumed from checkpoint: epoch {checkpoint['epoch']}")
        return start_epoch
    else:
        print("📝 No checkpoint found. Starting from scratch.")
        return 0

# -------------------- Training with checkpointing --------------------
def train():
    print("Loading dataset...")
    dataset = ColorizationRGBDataset("dataset", size=128)
    if len(dataset) == 0:
        print("❌ No images found! Check your dataset folder structure.")
        return
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

    # Device setup (CUDA -> MPS -> CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("🔥 Using CUDA GPU")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🔥 Using Apple M2 GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("🐢 Using CPU")

    # Initialize models
    gen = UNetGenerator().to(device)
    disc = PatchGANDiscriminator().to(device)

    # Loss and optimizers
    criterion_GAN = nn.BCELoss()
    criterion_L1 = nn.L1Loss()
    optimizer_G = torch.optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Try to resume from latest checkpoint
    start_epoch = load_checkpoint("checkpoint_latest.pth", gen, disc, optimizer_G, optimizer_D, device)

    total_epochs = 20      # Train up to 20 epochs total
    save_interval = 5      # Save checkpoint every 5 epochs

    print(f"Starting training from epoch {start_epoch+1} to {total_epochs}...")
    for epoch in range(start_epoch, total_epochs):
        epoch_start = time.time()
        gen_loss_avg = 0.0
        disc_loss_avg = 0.0

        for i, (gray, color) in enumerate(loader):
            gray = gray.to(device)
            color = color.to(device)

            # Generator output
            fake_color = gen(gray)

            # --- Train Discriminator ---
            pred_real = disc(color)
            loss_D_real = criterion_GAN(pred_real, torch.ones_like(pred_real))
            pred_fake = disc(fake_color.detach())
            loss_D_fake = criterion_GAN(pred_fake, torch.zeros_like(pred_fake))
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            optimizer_D.zero_grad()
            loss_D.backward()
            optimizer_D.step()

            # --- Train Generator ---
            pred_fake = disc(fake_color)
            loss_G_GAN = criterion_GAN(pred_fake, torch.ones_like(pred_fake))
            loss_G_L1 = criterion_L1(fake_color, color) * 100
            loss_G = loss_G_GAN + loss_G_L1
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            gen_loss_avg += loss_G.item()
            disc_loss_avg += loss_D.item()

        epoch_time = time.time() - epoch_start
        avg_gen = gen_loss_avg / len(loader)
        avg_disc = disc_loss_avg / len(loader)
        print(f"Epoch {epoch+1:2d} | G_loss: {avg_gen:.4f} | D_loss: {avg_disc:.4f} | Time: {epoch_time:.2f}s")

        # Save checkpoint every 'save_interval' epochs
        if (epoch + 1) % save_interval == 0:
            save_checkpoint(epoch, gen, disc, optimizer_G, optimizer_D, avg_gen, f"checkpoint_epoch_{epoch+1}.pth")
            save_checkpoint(epoch, gen, disc, optimizer_G, optimizer_D, avg_gen, "checkpoint_latest.pth")

    # Final save
    save_checkpoint(total_epochs-1, gen, disc, optimizer_G, optimizer_D, avg_gen, "pix2pix_generator_final.pth")
    torch.save(gen.state_dict(), "pix2pix_generator.pth")
    print("✅ Training complete. Final model saved as pix2pix_generator.pth")

if __name__ == "__main__":
    train()