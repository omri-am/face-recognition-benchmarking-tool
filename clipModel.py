from FacesBenchmarkUtils import *
import importlib.util
import subprocess
import sys

def install_clip():
    spec = importlib.util.find_spec("clip")
    if spec is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/CLIP.git"])
    else:
        print("CLIP is already installed")

install_clip()

import clip

class CLIPModel(BaseModel):
    def __init__(self, name: str, version="ViT-B/32"):
        self.version = version
        super().__init__(name=name)

    def _build_model(self):
        self.model, self.preprocess = clip.load(self.version, device=self.device)
        self.model.eval()

    def get_output(self, image_tensor):
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor.to(self.device))
        return image_features

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return self.preprocess(image).unsqueeze(0)
