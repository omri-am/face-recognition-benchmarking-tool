import importlib.util
import torch
from torch import nn
from abc import ABC, abstractmethod
from torchvision import models
from PIL import Image

def install_clip():
    spec = importlib.util.find_spec("clip")
    if spec is None:
        importlib.util.module_from_spec(importlib.util.spec_from_file_location("clip", "https://github.com/openai/CLIP.git"))
    else:
        print("CLIP is already installed")

install_clip()

import clip

class BaseModel(ABC):
    def __init__(self, model_path=None, extract_layer=None):
        self.hook_output = None
        self.extract_layer = extract_layer
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_identities = self._set_num_identities() if model_path else None
        self._build_model()
        if model_path:
            self._load_model()
        self.to()
        self._register_hook()

    def _set_num_identities(self):
        checkpoint = torch.load(self.model_path, map_location=self.device)
        if 'state_dict' in checkpoint:
            last_key = list(checkpoint['state_dict'].keys())[-1]
            return checkpoint['state_dict'][last_key].shape[0]
        else:
            last_key = list(checkpoint.keys())[-1]
            return checkpoint[last_key].shape[0]

    @abstractmethod
    def _build_model(self):
        pass

    def _load_model(self):
        if isinstance(self.model, nn.Module):
            checkpoint = torch.load(self.model_path, map_location=self.device)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                if isinstance(self.model, nn.DataParallel):
                    state_dict = self.model.module.state_dict()
                self.model.load_state_dict(state_dict)
                self._register_hook()
                self.model.eval()

    def _register_hook(self):
        if self.extract_layer is not None:
            for idx, layer in enumerate(self.model.modules()):
                if idx == self.extract_layer - 1:
                    layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.hook_output = output

    def hook_fn_input(self, module, input, output):
        self.hook_output = input[0]

    @abstractmethod
    def get_output(self, input):
        pass

    def to(self):
        if self.model:
            self.model.to(self.device)

## VGG (includes parallel)

class Vgg16ModelFC7(BaseModel):
    def __init__(self, model_path, extract_layer=34):
        super().__init__(model_path=model_path, extract_layer=extract_layer)

    def _build_model(self):
        model = models.vgg16(weights=None)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(int(num_features), int(self.num_identities))
        model.features = torch.nn.DataParallel(model.features)
        self.model = model

    def get_output(self, input):
        input = input.to(self.device)
        self.model(input)
        out = self.hook_output
        out = out.detach().cpu()
        out = out.reshape(1, -1)
        return out

## Clip

class CLIPModel(BaseModel):
    def __init__(self, version = "ViT-B/32"):
        self.version = version
        super().__init__()

    def _build_model(self):
        self.model, self.preprocess = clip.load(self.version, device=self.device)
        self.model.eval()

    def get_output(self, image_tensor):
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor.to(self.device))
        return image_features

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')  # Ensure the image is in RGB format
        return self.preprocess(image).unsqueeze(0)

## Dino
class DinoModel(BaseModel):
    def __init__(self, version = 'dino_vits16'):
        self.version = version
        super().__init__(model_path=model_path, extract_layer=extract_layer)

    def _build_model(self):
        self.model = torch.hub.load('facebookresearch/dino:main', version)

    def get_output(self, input):
        input = input.to(self.device)
        with torch.no_grad():
            out = self.model(input)
        out = out.detach().cpu().reshape(1, -1)
        return out