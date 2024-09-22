from FacesBenchmarkUtils import *
from torchvision import models
from torchvision.models import VGG16_Weights
import torch.nn as nn

class Vgg16Model(BaseModel):
    def __init__(self, name, weights_path=None, extract_layers='classifier.3', preprocess_function=None):
        super().__init__(name=name, weights_path=weights_path, extract_layers=extract_layers, preprocess_function=preprocess_function)

    def _build_model(self):
        model = models.vgg16(weights=VGG16_Weights.DEFAULT if self.weights_path is None else None)
        
        if self.num_identities is not None:
            # Modify the classifier to match the number of identities (classes)
            num_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_features, self.num_identities)
        else:
            # Default VGG16: the classifier outputs 1000 classes (ImageNet)
            self.num_identities = model.classifier[6].out_features
        
        self.model = model

    def _forward(self, input_tensor):
        # Ensure input_tensor has batch dimension
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
        return self.model(input_tensor)

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return self.preprocess(image)
