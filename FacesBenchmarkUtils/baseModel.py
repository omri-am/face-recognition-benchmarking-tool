from abc import ABC, abstractmethod
import torch
from torch import nn
import torchvision.transforms as transforms

class BaseModel(ABC):
    def __init__(self, name: str, weights_path: str = None, extract_layer: int = None, preprocess_function = None):
        self.set_preprocess_function(preprocess_function)
        self.hook_output = None
        self.name = name
        self.extract_layer = extract_layer
        self.weights_path = weights_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_identities = self._set_num_identities() if weights_path else None
        self._build_model()
        if weights_path:
            self._load_model()
        self.to()
        self._register_hook()

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
        state_dict = state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)

        if torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)
        self.to()
        self.model.eval()

    def _register_hook(self):
        if self.extract_layer is not None:
            for idx, layer in enumerate(self.model.modules()):
                if idx == self.extract_layer - 1:
                    layer.register_forward_hook(self.hook_fn)
                    break

    def hook_fn(self, module, input, output):
        self.hook_output = output

    def hook_fn_input(self, module, input, output):
        self.hook_output = input[0]

    @abstractmethod
    def get_output(self, image_tensor):
        pass

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

