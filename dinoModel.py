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

from transformers import AutoImageProcessor, AutoModel

class DinoModel(BaseModel):
    def __init__(self, name: str, version='facebook/dino-vitb8'):
        self.version = version
        super().__init__(name = name)

    def _build_model(self):
        self.model = AutoModel.from_pretrained(self.version)
        self.processor = AutoImageProcessor.from_pretrained(self.version)
        self.model.to(self.device)
        self.model.eval()

    def get_output(self, input_image):
        inputs = self.processor(images=input_image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        return outputs[0]

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)
        pixel_values = (pixel_values - pixel_values.min()) / (pixel_values.max() - pixel_values.min())

        return pixel_values