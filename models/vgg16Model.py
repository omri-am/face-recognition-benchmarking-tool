from facesBenchmarkUtils.baseModel import *
from torchvision import models
import torch.nn as nn

class Vgg16Model(BaseModel):
    def __init__(self, name, weights_path=None, extract_layers='classifier.3', preprocess_function=None):
        super().__init__(name=name, weights_path=weights_path, extract_layers=extract_layers, preprocess_function=preprocess_function)

    def _build_model(self):
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT if self.weights_path is None else None)
        
        if self.num_identities is not None:
            num_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_features, self.num_identities)
        else:
            self.num_identities = model.classifier[6].out_features
        
        self.model = model

    def _forward(self, input_tensor):
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
        return self.model(input_tensor)

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return self.preprocess(image)
