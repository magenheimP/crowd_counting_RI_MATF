import torch
import torch.nn as nn
import torch.optim as optim
from model import CSRNet  # your model.py
from pathlib import Path
from PIL import Image
import h5py
import cv2
from torchvision import transforms

# ---------------------------
# Settings
# ---------------------------
device = torch.device("cpu")  # CPU training
batch_size = 300
epochs = 300
lr = 1e-6

# Path to your dataset
img_folder = Path("/home/petar/Downloads/Shanghai/part_A_final/train_data/images")
gt_folder = Path("/home/petar/Downloads/Shanghai/part_A_final/train_data/ground_truth")

# Transform for images
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# ---------------------------
# Helper functions
# ---------------------------
def load_data(img_path):
    """Load image and corresponding density map (.h5)"""
    img = Image.open(img_path).convert('RGB')
    gt_path = gt_folder / img_path.name.replace('.jpg', '.h5')
    
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    with h5py.File(gt_path, 'r') as f:
        target = f['density'][:].astype('float32')

    # Resize density map to match CSRNet output (1/8)
    h, w = target.shape
    target = cv2.resize(target, (w // 8, h // 8), interpolation=cv2.INTER_CUBIC)
    target *= 64  # scale density

    return img, target

def load_dataset(batch_size, target_img_size=(512, 512)):
    """
    Load batch_size images and targets, resize all to the same size.
    target_img_size: tuple (H, W) for images (must be divisible by 8 for CSRNet)
    """
    img_paths = list(img_folder.glob("*.jpg"))
    imgs = []
    targets = []
    for path in img_paths[:batch_size]:  # load only batch_size images for now
        img, target = load_data(path)

        # Resize image
        img = img.resize(target_img_size, Image.BILINEAR)
        # Resize target to 1/8 of image size
        target_h, target_w = target_img_size[0] // 8, target_img_size[1] // 8
        target = cv2.resize(target, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        target *= 64

        
        img = transform(img)

        target = torch.from_numpy(target).unsqueeze(0)  # shape [1,H,W]

        imgs.append(img)
        targets.append(target)

    return torch.stack(imgs), torch.stack(targets)


# ---------------------------
# Model and optimizer
# ---------------------------
model = CSRNet(load_weights=False).to(device)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.95, weight_decay=5e-4)

# ---------------------------
# Training loop
# ---------------------------
model.train()

for epoch in range(epochs):
    imgs, targets = load_dataset(batch_size)
    imgs, targets = imgs.to(device), targets.to(device)

    optimizer.zero_grad()
    outputs = model(imgs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

print("Training complete on CPU with real images and ground truth.")

# After training
torch.save(model.state_dict(), "csrnet_weights.pth")
print("Model weights saved!")

