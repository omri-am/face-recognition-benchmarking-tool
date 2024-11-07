from facesBenchmarkUtils.baseModel import *
from torchvision import models
import torch.nn as nn

class Vgg16Model(BaseModel):
    """
    A VGG16 model implementation for face recognition tasks.

    This class initializes a VGG16 model, optionally loading pre-trained weights.
    It allows extraction of specific layers and provides methods for preprocessing
    images and forwarding inputs through the model.

    Attributes
    ----------
    model_name : str
        The name of the model.
    weights_file_path : str or None
        Path to the model's weights file (.pth extention). If None, default pre-trained weights are used.
    extract_layers : str or list of str
        Layer(s) from which to extract outputs.
    preprocess_function : callable or None
        Function to preprocess input images.
    num_identities : int or None
        Number of identities (classes) in the model, set if weights are loaded.
    model : torch.nn.Module
        The VGG16 neural network model.
    device : torch.device
        The device (CPU or GPU) on which the model is placed.
    hook_outputs : dict
        Dictionary to store outputs from hooked layers.
    """

    def __init__(
        self,
        model_name: str,
        weights_file_path: Optional[str] = None,
        layers_to_extract: Optional[Union[str, List[str]]] = 'classifier.3',
        preprocess_function: Optional[Callable[[Any], Any]] = None
    ):
        """
        Initializes the Vgg16Model.

        Parameters
        ----------
        model_name : str
            The name of the model.
        weights_file_path : str or None, optional
            Path to the model's weights file. If None, default pre-trained weights are used.
        layers_to_extract : str or list of str, optional
            Layer(s) from which to extract outputs. Defaults to 'classifier.3'.
        preprocess_function : callable or None, optional
            Function to preprocess input images. If None, no additional preprocessing is applied.
        """
        super().__init__(
            model_name=model_name,
            weights_file_path=weights_file_path,
            layers_to_extract=layers_to_extract,
            preprocess_function=preprocess_function
        )

    def _build_model(self):
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT if self.weights_file_path is None else None)
        
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
