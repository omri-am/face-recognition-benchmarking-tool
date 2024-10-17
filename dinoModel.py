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

from transformers import AutoImageProcessor, Dinov2Model

class DinoModel(BaseModel):
    def __init__(self, name: str, version='facebook/dinov2-base'):
        self.version = version
        super().__init__(name=name)

    def _build_model(self):
        self.model = Dinov2Model.from_pretrained(self.version)
        self.processor = AutoImageProcessor.from_pretrained(self.version)
        self.model.to(self.device)
        self.model.eval()

    def _forward(self, input_tensor):
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
        output = self.model(pixel_values=input_tensor)
        # Return the last_hidden_state as default output
        return output.last_hidden_state

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        return inputs['pixel_values'][0]
