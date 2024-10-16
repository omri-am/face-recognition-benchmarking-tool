from abc import ABC, abstractmethod
import torch
from torch import nn
import torchvision.transforms as transforms
from PIL import Image

class BaseModel(ABC):
    def __init__(self, name: str, weights_path: str = None, extract_layers = None, preprocess_function = None):
        self.set_preprocess_function(preprocess_function)
        self.hook_outputs = {}
        self.name = name
        self.extract_layers = extract_layers if isinstance(extract_layers, list) else ([extract_layers] if extract_layers else [])
        self.weights_path = weights_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_identities = self._set_num_identities() if weights_path else None
        self._build_model()
        if weights_path:
            self._load_model()
        self.to()
        self.model.eval()
        self._register_hooks()

    def _set_num_identities(self):
        checkpoint = torch.load(self.weights_path, map_location=self.device)
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
        checkpoint = torch.load(self.weights_path, map_location=self.device)
        state_dict = checkpoint.get('state_dict', checkpoint)
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.to()
        self.model.eval()

    def print_layer_names(self, simplified=False):
        layers = dict(self.model.named_modules())
        for name, info in layers.items():
            print(f'{name}\n{info}\n' if not simplified else name)

    def _register_hooks(self):
        if self.extract_layers:
            for layer_name in self.extract_layers:
                layer = self._get_layer(layer_name)
                if layer:
                    layer.register_forward_hook(self._get_hook_fn(layer_name))

    def _get_hook_fn(self, layer_name):
        def hook_fn(module, input, output):
            self.hook_outputs[layer_name] = output
        return hook_fn
    
    def _get_layer(self, layer_name):
        """
        Get the layer by name (e.g., 'features.30', 'classifier.3').
        This will handle complex architectures like ResNet, VGG, etc.
        """
        if layer_name is None:
            return
        modules = dict(self.model.named_modules())
        if layer_name in modules:
            return modules[layer_name]
        else:
            raise ValueError(f"Layer {layer_name} not found in the model.")

    @abstractmethod
    def _forward(self, input_tensor):
        pass

    def get_output(self, input_tensor):
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
                
    def to(self):
        if self.model:
            self.model.to(self.device)

    def set_preprocess_function(self, preprocess_function):
        """
        Sets the preprocessing function for images. Uses a default function if none is provided.
        """
        if preprocess_function is None:
            self.preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.preprocess = preprocess_function