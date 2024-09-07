from FacesBenchmarkUtils import *
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

from transformers import AutoModel, AutoImageProcessor

class DinoModel(BaseModel):
    def __init__(self, name: str, version='facebook/dinov2-base', extract_layer: int = -1):
        self.version = version
        super().__init__(name=name, extract_layer=extract_layer)

    def _build_model(self):
        self.model = AutoModel.from_pretrained(self.version)
        self.processor = AutoImageProcessor.from_pretrained(self.version)
        self.model.to(self.device)
        self.model.eval()

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs['pixel_values'].unsqueeze(0)

    def get_output(self, input_tensor):
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            output = self.model(pixel_values=input_tensor)
        flattened_output = output.last_hidden_state.view(output.last_hidden_state.size(0), -1)
        return flattened_output
