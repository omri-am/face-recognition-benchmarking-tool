from FacesBenchmarkUtils import *
from torchvision import models
from torchvision.models import VGG16_Weights
import torch.nn as nn

class Vgg16Model(BaseModel):
    """
    A class representing a VGG16-based neural network model for face-related experiments.
    
    This model is a subclass of the abstract `BaseModel` class and allows for flexible usage
    with either pre-trained weights (e.g., ImageNet) or custom weights. The model supports
    the extraction of intermediate layer outputs for further analysis.

    Attributes:
    -----------
    name : str
        The name of the model instance.
    weights_path : str, optional
        The path to the custom model weights. If not provided, the default VGG16 with ImageNet
        pre-trained weights is used.
    extract_layer : str, optional
        The name of the layer from which to extract features. Defaults to 'classifier.3' for VGG16.
    preprocess_function : callable, optional
        A custom preprocessing function for the input images. If not provided, a default
        preprocessing pipeline is used.

    Methods:
    --------
    _build_model():
        Builds the VGG16 model architecture. If `weights_path` is provided, the model's output
        layer is modified to match the number of identities (classes). Otherwise, the default
        VGG16 architecture (with 1000 classes) is used.
    get_output(image_tensor):
        Processes the input image tensor and returns the extracted output from the specified layer.
    """
    
    def __init__(self, name: str, weights_path: str=None, extract_layer: str = 'classifier.3', preprocess_function=None):
        """
        Initializes the Vgg16Model instance.

        Parameters:
        -----------
        name : str
            The name of the model instance.
        weights_path : str, optional
            The path to the custom model weights. If not provided, the default VGG16 with ImageNet
            pre-trained weights is used.
        extract_layer : str, optional
            The name of the layer from which to extract features. Defaults to 'classifier.3' for VGG16.
        preprocess_function : callable, optional
            A custom preprocessing function for the input images. If not provided, a default
            preprocessing pipeline is used.
        """
        super().__init__(name=name, weights_path=weights_path, extract_layer=extract_layer, preprocess_function=preprocess_function)

    def _build_model(self):
        """
        Builds the VGG16 model architecture.

        If `weights_path` is provided, the model's classifier layer is modified to output a number
        of classes that matches the number of identities specified by the custom weights.
        If no `weights_path` is provided, the default VGG16 architecture (with 1000 classes for ImageNet)
        is used.
        """
        model = models.vgg16(weights=VGG16_Weights.DEFAULT if self.weights_path is None else None)
        
        if self.num_identities is not None:
            # Modify the classifier to match the number of identities (classes)
            num_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_features, self.num_identities)
        else:
            # Default VGG16: the classifier outputs 1000 classes (ImageNet)
            self.num_identities = model.classifier[6].out_features
        
        self.model = model

    def get_output(self, image_tensor):
        """
        Processes the input image tensor and extracts features from the specified layer.

        Parameters:
        -----------
        image_tensor : torch.Tensor
            A 4D image tensor (batch size, channels, height, width) representing the input image.
        
        Returns:
        --------
        torch.Tensor
            The output features from the specified layer, after processing the input image.
        """
        image_tensor = image_tensor.to(self.device)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        with torch.no_grad():
            _ = self.model(image_tensor)
            out = self.hook_output
            out = out.detach().cpu()
            return out.view(1, -1)
