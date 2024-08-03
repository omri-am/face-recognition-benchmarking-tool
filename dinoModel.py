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

from transformers import ViTImageProcessor, ViTModel

class DinoModel(BaseModel):
    def __init__(self, name: str, version='facebook/dino-vitb8'):
        self.version = version
        super().__init__(name = name)

    def _build_model(self):
        self.processor = ViTImageProcessor.from_pretrained(self.version)
        self.model = ViTModel.from_pretrained(self.version)
        self.model.eval()

    def get_output(self, input_image):
        inputs = self.processor(images=input_image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return image