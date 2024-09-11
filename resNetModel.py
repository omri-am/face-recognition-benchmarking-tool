from transformers import DetrImageProcessor, DetrForObjectDetection
from FacesBenchmarkUtils import *

class ResNetModel(BaseModel):
    def __init__(self, name: str):
        super().__init__(name=name, extract_layer=None)

    def _build_model(self):
        self.model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        self.processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
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