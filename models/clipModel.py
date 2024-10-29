from facesBenchmarkUtils.baseModel import *
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
    """
    A CLIP model implementation for face recognition tasks.

    This class initializes a CLIP model with a specified version, handles image
    preprocessing, and provides methods for forwarding inputs through the model.

    Attributes
    ----------
    name : str
        The name of the model.
    version : str
        The version identifier for the CLIP model.
    model : torch.nn.Module
        The CLIP visual neural network model.
    preprocess : callable
        Function to preprocess input images.
    device : torch.device
        The device (CPU or GPU) on which the model is placed.
    hook_outputs : dict
        Dictionary to store outputs from hooked layers.
    """

    def __init__(
        self, 
        name: str, 
        version: str = "ViT-B/32"
    ):
        """
        Initializes the CLIPModel.

        Parameters
        ----------
        name : str
            The name of the model.
        version : str, optional
            The version identifier for the CLIP model. Defaults to "ViT-B/32".
        """
        self.version = version
        super().__init__(name=name)

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
