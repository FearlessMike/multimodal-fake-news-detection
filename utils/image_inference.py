import torch
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

# Load pretrained EfficientNet
model = models.efficientnet_b0(weights="DEFAULT")
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

def predict_image(image: Image.Image) -> float:
    """
    Returns an image credibility score in [0, 1].
    Higher = more suspicious / less confident.
    """
    image = image.convert("RGB")
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)

    # Entropy-based uncertainty score
    entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)

    # Normalize entropy to [0,1]
    max_entropy = torch.log(torch.tensor(probs.shape[1], dtype=torch.float))
    score = (entropy / max_entropy).item()

    return score
