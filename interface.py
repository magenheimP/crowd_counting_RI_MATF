from model import CSRNet
import torch
from PIL import Image
from torchvision import transforms
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm as CM


device = torch.device("cpu")  # or "cuda" if GPU available

# Create a model instance
model = CSRNet(load_weights=False).to(device)

#loading the pre-trained weights and a best model for comparison
best_pretrained_mozel = CSRNet(load_weights=True).to(device)
checkpoint = torch.load("PartAmodel_best.pth.tar", map_location=device, weights_only=False)

if "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

best_pretrained_mozel.load_state_dict(state_dict)

best_pretrained_mozel.eval()
print("✅ Pretrained CSRNet weights loaded successfully. Best model ready to use!")




# Load the saved weights
model.load_state_dict(torch.load("csrnet_weights.pth", map_location=device))
model.eval()  # important for inference



# Load and preprocess image
img_path = "/home/petar/Downloads/Shanghai/part_A_final/test_data/images/IMG_1.jpg"
img = Image.open(img_path).convert('RGB')

# Resize image if needed (e.g., multiples of 8)
# img = img.resize((512, 512), Image.BILINEAR)

# Transform to tensor
transform = transforms.Compose([
    transforms.ToTensor()
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225])
])
img_tensor = transform(img).unsqueeze(0)  # add batch dimension [1,3,H,W]

# Run inference
with torch.no_grad():
    output = best_pretrained_mozel(img_tensor)
    
# output is the predicted density map
density_map = output.squeeze(0).squeeze(0).numpy()  # shape [H//8, W//8]
# Optional: sum to get predicted count
predicted_count = density_map.sum()
print("Predicted count:", predicted_count)


# Display the image and the output next to each other
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# Display the original image
axes[0].imshow(Image.open(img_path).convert('RGB'))
axes[0].set_title('Original Image')
axes[0].axis('off')

# Display the model output (density map)
axes[1].imshow(output.detach().cpu().numpy().squeeze(), cmap=CM.RdBu)
axes[1].set_title('Density Map')
axes[1].axis('off')

plt.tight_layout()
plt.show()

density_map = output.squeeze(0).squeeze(0).detach().cpu().numpy()
predicted_count = density_map.sum()
print("Predicted count:", predicted_count)
