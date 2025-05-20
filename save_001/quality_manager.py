import torch
import torchvision.transforms as transforms
from PIL import Image
import yaml

device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
model = torch.hub.load(repo_or_dir="miccunifi/ARNIQA", source="github", model="ARNIQA",
                       regressor_dataset="kadid10k")    # You can choose any of the available datasets
model.eval().to(device)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img_path = "project-02/yenidenemem_few_shot/face_teslim/face_db/Human/Classes/1002/images/face_f16_d0_20250515_134928_894330.png"
img = Image.open(img_path).convert("RGB")

img_ds = transforms.Resize((img.size[1] // 2, img.size[0] // 2))(img)
img = preprocess(img).unsqueeze(0).to(device)
img_ds = preprocess(img_ds).unsqueeze(0).to(device)

with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
    score = model(img, img_ds, return_embedding=False, scale_score=True)

print(f"Image quality score: {score.item()}")

class ImageValdator():
    def __init__(self, config_path):    
        try:
            with open(config_path,'r',encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            print("File not found")
            self.config = {}
        except yaml.YAMLError as e:
            print("Error: Yaml error")
            
        # print(self.config)
        self.device = self.config.get("default_device", "cpu")
        self.model_config = self.config.get("quality_assessment", {})
        self.model_name = self.model_config.get("method_name", 'arniqa') 
                        
if __name__=="__main__":
    ImageValdator("/home/earsal@ETE.local/Desktop/codes/seeing-any/project-02/yenidenemem_few_shot/face_teslim/config.yaml")