from FacesBenchmarkUtils import *
from torchvision import models
import torch.nn as nn

class Vgg16Model(BaseModel):
    def __init__(self, name: str, weights_path: str, extract_layer: int = 34, preprocess_function=None):
        super().__init__(name=name, weights_path=weights_path, extract_layer=extract_layer, preprocess_function=preprocess_function)

    def _build_model(self):
        model = models.vgg16(weights=None)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(int(num_features), int(self.num_identities))
        self.model = model

    def get_output(self, image_tensor):
        image_tensor = image_tensor.to(self.device)
        image_tensor = image_tensor.unsqueeze(0)
        with torch.no_grad():
            _ = self.model(image_tensor)
            out = self.hook_output
            out = out.detach().cpu()
            return out.view(1, -1)
