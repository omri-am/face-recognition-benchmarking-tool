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
    def __init__(self, name, version="ViT-B/32", extract_layers=None):
        self.version = version
        super().__init__(name=name, extract_layers=extract_layers)

    def _build_model(self):
        self.model, self.preprocess = clip.load(self.version, device=self.device)
        self.model = self.model.visual
        self.model.eval()
        self.model.float()

    def _forward(self, input_tensor):
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
        return self.model(input_tensor)

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return self.preprocess(image)
