from facesBenchmarkUtils.baseModel import *
import subprocess
import sys

def pipinstall(command):    
    try:
        subprocess.check_call(command)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while installing: {e}")

transformers_command = [
        sys.executable, '-m', 'pip', 'install',
        '--upgrade', 'tqdm', '-q'
    ]
tqdm_command = [sys.executable, '-m', 'pip', 'install',
        '--upgrade', 'transformers', '-q'
    ]
pipinstall(transformers_command)
pipinstall(tqdm_command)

from transformers import AutoImageProcessor, Dinov2Model

class DinoModel(BaseModel):
    """
    A DINO model implementation for face recognition tasks.

    This class initializes a DINO model using the specified version, handles image
    preprocessing, and provides methods for forwarding inputs through the model.

    Attributes
    ----------
    name : str
        The name of the model.
    version : str
        The version identifier for the DINO model.
    model : torch.nn.Module
        The DINO neural network model.
    processor : transformers.AutoImageProcessor
        The image processor for preparing inputs.
    device : torch.device
        The device (CPU or GPU) on which the model is placed.
    hook_outputs : dict
        Dictionary to store outputs from hooked layers.
    """

    def __init__(
        self, 
        model_name: str, 
        version: str = 'facebook/dinov2-base',
        layers_to_extract: Optional[Union[str, List[str]]] = None
    ):
        """
        Initializes the DinoModel.

        Parameters
        ----------
        model_name : str
            The name of the model.
        version : str, optional
            The version identifier for the DINO model. Defaults to 'facebook/dinov2-base'.
        """
        self.version = version
        super().__init__(model_name=model_name, layers_to_extract=layers_to_extract)

    def _build_model(self):
        self.model = Dinov2Model.from_pretrained(self.version)
        self.processor = AutoImageProcessor.from_pretrained(self.version)
        self.model.to(self.device)
        self.model.eval()

    def _forward(self, input_tensor):
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
        output = self.model(pixel_values=input_tensor)
        return output.last_hidden_state

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs['pixel_values'][0]
