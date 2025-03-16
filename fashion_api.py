from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import io
import os

app = FastAPI()

# Map class indices to class names:
class_mappings = {
    'articleType': {0: 'Backpacks',
                 1: 'Belts',
                 2: 'Bra',
                 3: 'Briefs',
                 4: 'Capris',
                 5: 'Caps',
                 6: 'Casual Shoes',
                 7: 'Clutches',
                 8: 'Deodorant',
                 9: 'Dresses',
                 10: 'Earrings',
                 11: 'Flats',
                 12: 'Flip Flops',
                 13: 'Formal Shoes',
                 14: 'Handbags',
                 15: 'Heels',
                 16: 'Innerwear Vests',
                 17: 'Jackets',
                 18: 'Jeans',
                 19: 'Kurtas',
                 20: 'Kurtis',
                 21: 'Leggings',
                 22: 'Lip Gloss',
                 23: 'Lipstick',
                 24: 'Nail Polish',
                 25: 'Necklace and Chains',
                 26: 'Night suits',
                 27: 'Nightdress',
                 28: 'Pendant',
                 29: 'Perfume and Body Mist',
                 30: 'Sandals',
                 31: 'Sarees',
                 32: 'Shirts',
                 33: 'Shorts',
                 34: 'Socks',
                 35: 'Sports Shoes',
                 36: 'Sunglasses',
                 37: 'Sweaters',
                 38: 'Sweatshirts',
                 39: 'Ties',
                 40: 'Tops',
                 41: 'Track Pants',
                 42: 'Trousers',
                 43: 'Trunk',
                 44: 'Tshirts',
                 45: 'Tunics',
                 46: 'Wallets',
                 47: 'Watches'},
 'baseColour': {0: 'Black',
                1: 'Blue',
                2: 'Brown',
                3: 'Green',
                4: 'Grey',
                5: 'Multi',
                6: 'Orange',
                7: 'Pink',
                8: 'Purple',
                9: 'Red',
                10: 'White',
                11: 'Yellow'},
 'gender': {0: 'Men', 1: 'Unisex', 2: 'Women'},
 'season': {0: 'Fall', 1: 'Spring', 2: 'Summer', 3: 'Winter'}
}

# ======= MODEL SETUP =======
bb1 = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
bb1 = nn.Sequential(*list(bb1.children())[:-1], nn.Flatten())

class FashionProductClassifier(nn.Module):
    def __init__(self, backbone, backbone_features, num_classes, dropout_prob=0.5):
        super(FashionProductClassifier, self).__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(p=dropout_prob)
        self.gender_head = nn.Linear(backbone_features, num_classes["gender"])
        self.articleType_head = nn.Linear(backbone_features, num_classes["articleType"])
        self.baseColour_head = nn.Linear(backbone_features, num_classes["baseColour"])
        self.season_head = nn.Linear(backbone_features, num_classes["season"])

    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        gender_out = self.gender_head(features)
        articleType_out = self.articleType_head(features)
        baseColour_out = self.baseColour_head(features)
        season_out = self.season_head(features)
        return {
            "baseColour": baseColour_out,
            "articleType": articleType_out,
            "season": season_out,
            "gender": gender_out,
        }

num_classes = {"gender": 3, "articleType": 48, "baseColour": 12, "season": 4}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FashionProductClassifier(bb1, 1280, num_classes)
model = model.to(device)

# ======= LOAD CHECKPOINT =======
checkpoint_path = os.path.join(os.path.dirname(__file__), "checkpt_model.pth")
checkpoint = torch.load(checkpoint_path, map_location=device)
state_dict = checkpoint['model_state_dict']
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace("module.", "")
    new_state_dict[new_key] = v
model.load_state_dict(new_state_dict)
model.eval()

# ===== TRANSFORMS =====
resize_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
model_transform = transforms.Compose([transforms.ToTensor()])

# Create intermediate image folder
intermediate_folder = "intermediate_images"
os.makedirs(intermediate_folder, exist_ok=True)

# ===== ENDPOINT =====
@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    image = Image.open(io.BytesIO(img_bytes)).convert('RGB')

    # Apply transform + save intermediate image
    transformed = resize_transform(image)
    transformed_pil = transforms.ToPILImage()(transformed)
    intermediate_img_path = os.path.join(intermediate_folder, file.filename)
    transformed_pil.save(intermediate_img_path)

    # Open intermediate image again
    p_img = Image.open(intermediate_img_path).convert('RGB')
    input_tensor = model_transform(p_img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        preds = { 
            "gender": torch.argmax(outputs["gender"], dim=1).item(),
            "articleType": torch.argmax(outputs["articleType"], dim=1).item(),
            "baseColour": torch.argmax(outputs["baseColour"], dim=1).item(),
            "season": torch.argmax(outputs["season"], dim=1).item()
        }
    
    decoded_preds = {
        "gender": class_mappings["gender"][preds["gender"]],
        "articleType": class_mappings["articleType"][preds["articleType"]],
        "baseColour": class_mappings["baseColour"][preds["baseColour"]],
        "season": class_mappings["season"][preds["season"]]
    }

    return decoded_preds
