from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
from torch import nn
import torchvision.transforms as transforms
from PIL import Image

class BaseModel(ABC):
    """
    An abstract base class representing a neural network model in the benchmarking framework.
    All specific model classes should inherit from this class.

    Attributes
    ----------
    name : str
        The name of the model.
    weights_path : str or None
        Path to the model's weights file. If None, default weights are used.
    extract_layers : list of str
        List of layer names from which to extract outputs.
    preprocess : callable
        Function to preprocess input images.
    hook_outputs : dict
        Dictionary to store outputs from hooked layers.
    model : torch.nn.Module
        The neural network model.
    device : torch.device
        The device (CPU or GPU) on which the model is placed.
    num_identities : int or None
        Number of identities (classes) in the model, set if weights are loaded.
    """

    def __init__(
        self,
        name: str,
        weights_path: Optional[str] = None,
        extract_layers: Optional[Union[str, List[str]]] = None,
        preprocess_function: Optional[Callable[[Any], Any]] = None
    ) -> None:
        """
        Initializes the BaseModel instance.

        Parameters
        ----------
        name : str
            The name of the model.
        weights_path : str, optional
            Path to the model's weights file. If None, default weights are used.
        extract_layers : str or list of str, optional
            Layer name(s) from which to extract outputs.
        preprocess_function : callable, optional
            Function to preprocess input images. If None, a default preprocessing is used.
        """
        self.set_preprocess_function(preprocess_function)
        self.hook_outputs: Dict[str, Any] = {}
        self.name: str = name
        if isinstance(extract_layers, list):
            self.extract_layers: List[str] = extract_layers
        elif extract_layers:
            self.extract_layers: List[str] = [extract_layers]
        else:
            self.extract_layers = []
        self.weights_path: Optional[str] = weights_path
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_identities: Optional[int] = self._set_num_identities() if weights_path else None
        self.model: Optional[nn.Module] = None
        self._build_model()
        if weights_path:
            self._load_model()
        self.to()
        if self.model:
            self.model.eval()
        self._register_hooks()

    def _set_num_identities(self) -> int:
        """
        Determines the number of identities (classes) based on the loaded weights.

        Returns
        -------
        int
            Number of identities in the model.
        """
        checkpoint = torch.load(self.weights_path, map_location=self.device)
        if 'state_dict' in checkpoint:
            last_key = list(checkpoint['state_dict'].keys())[-1]
            return checkpoint['state_dict'][last_key].shape[0]
        else:
            last_key = list(checkpoint.keys())[-1]
            return checkpoint[last_key].shape[0]

    @abstractmethod
    def _build_model(self) -> None:
        """
        Abstract method to build the neural network model.
        Must be implemented by subclasses.
        """
        pass

    def _load_model(self) -> None:
        """
        Loads the model weights from the specified path.
        """
        checkpoint = torch.load(self.weights_path, map_location=self.device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        if self.model:
            self.model.load_state_dict(state_dict)

            if torch.cuda.device_count() > 1:
                self.model = nn.DataParallel(self.model)
            self.to()
            self.model.eval()

    def print_layer_names(self, simplified: bool = False) -> None:
        """
        Prints the names of all layers in the model.

        Parameters
        ----------
        simplified : bool, optional
            If True, prints only the layer names. If False, also prints layer details.
        """
        if self.model:
            layers = dict(self.model.named_modules())
            for name, info in layers.items():
                print(f'{name}\n{info}\n' if not simplified else name)

    def _register_hooks(self) -> None:
        """
        Registers forward hooks on specified layers to capture their outputs.
        """
        if self.extract_layers:
            for layer_name in self.extract_layers:
                layer = self._get_layer(layer_name)
                if layer:
                    layer.register_forward_hook(self._get_hook_fn(layer_name))

    def _get_hook_fn(self, layer_name: str) -> Callable[[nn.Module, Any, Any], None]:
        """
        Creates a hook function to capture the output of a layer.

        Parameters
        ----------
        layer_name : str
            The name of the layer to hook.

        Returns
        -------
        callable
            The hook function.
        """
        def hook_fn(module: nn.Module, input: Any, output: Any) -> None:
            self.hook_outputs[layer_name] = output
        return hook_fn

    def _get_layer(self, layer_name: str) -> Optional[nn.Module]:
        """
        Retrieves a layer from the model by its name.

        Parameters
        ----------
        layer_name : str
            The name of the layer.

        Returns
        -------
        torch.nn.Module or None
            The requested layer.

        Raises
        ------
        ValueError
            If the layer name is not found in the model.
        """
        if layer_name is None or self.model is None:
            return None
        modules = dict(self.model.named_modules())
        if layer_name in modules:
            return modules[layer_name]
        else:
            raise ValueError(f"Layer {layer_name} not found in the model.")

    @abstractmethod
    def _forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """
        Abstract method for the forward pass of the model.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The input tensor to the model.

        Returns
        -------
        torch.Tensor
            The output tensor from the model.

        Raises
        ------
        NotImplementedError
            Must be implemented in subclasses.
        """
        pass

    def get_output(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Runs the model on the input tensor and retrieves outputs from specified layers.

        Parameters
        ----------
        input_tensor : torch.Tensor
            The preprocessed input tensor.

        Returns
        -------
        dict
            A dictionary mapping layer names to their output tensors.
            If no hooks are registered, returns the default output.
        """
        self.hook_outputs = {}
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            output = self._forward(input_tensor)
            if self.hook_outputs:
                outputs = {
                    layer_name: out.detach().cpu().view(out.size(0), -1)
                    for layer_name, out in self.hook_outputs.items()
                }
            else:
                outputs = {'default': output.detach().cpu().view(output.size(0), -1)}
            return outputs

    def to(self) -> None:
        """
        Moves the model to the specified device (CPU or GPU).
        """
        if self.model:
            self.model.to(self.device)

    def set_preprocess_function(
        self, preprocess_function: Optional[Callable[[Any], Any]]
    ) -> None:
        """
        Sets the preprocessing function for input images.

        Parameters
        ----------
        preprocess_function : callable or None
            A function that preprocesses PIL images into tensors.
            If None, a default preprocessing function is used.
        """
        if preprocess_function is None:
            self.preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.preprocess = preprocess_function