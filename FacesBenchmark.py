import importlib.util
import subprocess
import sys
import os
import torch
from torch import nn
from abc import ABC, abstractmethod
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
from torchvision import models
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from datetime import date
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, pairwise
import matplotlib.pyplot as plt
import seaborn as sns


def install_clip():
    spec = importlib.util.find_spec("clip")
    if spec is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/openai/CLIP.git"])
    else:
        print("CLIP is already installed")

install_clip()

import clip

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
        if isinstance(self.model, nn.Module):
            checkpoint = torch.load(self.weights_path, map_location=self.device)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
                if isinstance(self.model, nn.DataParallel):
                    state_dict = self.model.module.state_dict()
                self.model.load_state_dict(state_dict)
                self._register_hook()
                self.model.eval()

    def _register_hook(self):
        if self.extract_layer is not None:
            for idx, layer in enumerate(self.model.modules()):
                if idx == self.extract_layer - 1:
                    layer.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.hook_output = output

    def hook_fn_input(self, module, input, output):
        self.hook_output = input[0]

    @abstractmethod
    def get_output(self, input):
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

class Vgg16Model(BaseModel):
    def __init__(self, name: str, weights_path: str, extract_layer: int=34, preprocess_function=None):
        super().__init__(name=name, weights_path=weights_path, extract_layer=extract_layer, preprocess_function=preprocess_function)

    def _build_model(self):
        model = models.vgg16(weights=None)
        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(int(num_features), int(self.num_identities))
        model.features = torch.nn.DataParallel(model.features)
        self.model = model

    def get_output(self, input):
        input = input.to(self.device)
        self.model(input)
        out = self.hook_output
        out = out.detach().cpu()
        out = out.reshape(1, -1)
        return out

class CLIPModel(BaseModel):
    def __init__(self, name: str, version="ViT-B/32"):
        self.version = version
        super().__init__(name=name)

    def _build_model(self):
        self.model, self.preprocess = clip.load(self.version, device=self.device)
        self.model.eval()

    def get_output(self, image_tensor):
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensor.to(self.device))
        return image_features

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return self.preprocess(image).unsqueeze(0)

class DinoModel(BaseModel):
    def __init__(self, name: str, version='facebook/dinov2-base'):
        self.version = version
        super().__init__(name = name)

    def _build_model(self):
        self.processor = AutoImageProcessor.from_pretrained(self.version)
        self.model = AutoModel.from_pretrained(self.version)
        self.model.eval()

    def get_output(self, input_image):
        inputs = self.processor(images=input_image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        return outputs.last_hidden_state

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        return image

class BaseTask(ABC):
    """
    A base class to represent a task. All task classes should inherit from this class.

    Attributes:
        name: The name of the task.
        images_path: The path where the images are stored.
        pairs_file_path: The path where the pairs file is stored.
        distance_metric: The distance metric to be used for evaluating the task.
    """
    def __init__(self, name: str, pairs_file_path: str, images_path: str, distance_metric=pairwise.cosine_distances):
        self.name = name
        self.pairs_file_path = pairs_file_path
        self.pairs_df = self.__load_file(pairs_file_path)
        self.images_path = self.__validate_path(images_path)
        self.distance_metric = self.__validate_distance_metric(distance_metric)

    def __to_float(self, x):
        if np.isscalar(x):
            return float(x)
        elif x.size == 1:
            return float(x.item())
        else:
            return float(x.ravel()[0])

    def __validate_distance_metric(self, user_func):
        try:
            rand_t1 = torch.rand((2, 3))
            rand_t2 = torch.rand((2, 3))
            result = user_func(rand_t1, rand_t2)
            self.__to_float(result)
        except Exception as e:
            raise Exception("Distance metric is not valid!") from e

        try:
            self.__to_float(user_func(rand_t1, rand_t2))
        except Exception:
            print(f"WARNING! The distance function does not return a scalar or an array. This could potentially affect computing. Please consider changing the function.")
        return lambda x, y: self.__to_float(user_func(x, y))


    def __validate_path(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError("File Not Found! Please provide the full path of the file.")
        return path

    def __load_file(self, pairs_file_path):
        self.__validate_path(pairs_file_path)
        try:
            pairs_pd = pd.read_csv(pairs_file_path)
            if not {'img1', 'img2'}.issubset(pairs_pd.columns):
                raise Exception("img1 and img2 columns are required!")
            return pairs_pd
        except Exception as e:
            raise e

    @abstractmethod
    def compute_task_performance(self, distances):
        pass

class AccuracyTask(BaseTask):
    """
    A task that evaluates the accuracy of the model.
    Attributes:
        true_label: Column name in the pairs file loaded, indicating whether the answer is correct.
    """
    def __init__(self, name: str, pairs_file_path: str, images_path: str, distance_metric, true_label: str = 'truth'):
        super().__init__(name=name, pairs_file_path=pairs_file_path, images_path=images_path, distance_metric=distance_metric)
        self.true_label = true_label

    def compute_task_performance(self, pairs_df_with_calc):
        similarity = pairs_df_with_calc['model_computed_distance'].apply(lambda x: 1-float(x))

        y_true = pairs_df_with_calc[self.true_label]
        auc = roc_auc_score(y_true, similarity)

        fpr, tpr, thresholds = roc_curve(y_true, similarity)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        y_pred = (similarity > optimal_threshold).astype(int)
        accuracy = accuracy_score(y_true, y_pred)

        return pd.DataFrame({"Accuracy": accuracy,
                             "Optimal Threshold": optimal_threshold,
                             "AUC": auc})

class CorrelationTask(BaseTask):
    """
    A task that evaluates the correlation between the model's distance metric and the true labels.
    Attributes:
        correlation_metric: The correlation metric to be used for evaluating the task.
    """
    def __init__(self, name: str, pairs_file_path: str, images_path: str, distance_metric, correlation_metric = np.corrcoef):
        super().__init__(name=name, pairs_file_path=pairs_file_path, images_path=images_path, distance_metric=distance_metric)

    def compute_task_performance(self, distances):
        correlation = self.correlation_metric(distances, self.pairs_df['distance'])[0, 1]
        return pd.DataFrame({"Correlation Score": correlation})

class MultiModelTaskManager():
    """
    A helper class to work with multiple neural network models and tasks.

    Attributes:
        models: A dictionary of neural network models.
        tasks: Dictionary storing tasks information.
        model_task_distances_dfs: Dictionary storing distances computed between images, by task and model.
        images_tensors: Dictionary storing image tensors, by model.
        tasks_performance_dfs: Dictionary storing task results as dataframes, by task type.
    """
    def __init__(self, models, tasks):
        self.tasks = {}
        self.add_tasks(tasks)
        self.models = {}
        self.add_models(models)
        self.model_task_distances_dfs = {model: {task_name: task.pairs_df for task_name, task in self.tasks.items()} for model in self.models}
        self.images_tensors = {model: {} for model in self.models}
        self.tasks_performance_dfs = {task: None for task in self.tasks}


    def __repr__(self):
        return (f"MultiModelTaskManager(models={list(self.models.keys())}, "
                f"tasks={list(self.tasks.keys())}")

    def __add_distinct_value_to_dict(self, dict_obj, key, value):
        if key in dict_obj.keys():
            del dict_obj[key]
        dict_obj[key] = value

    def add_tasks(self, tasks):
        """
        Adds tasks to the manager. Tasks should be subclasses of BaseTask.
        """

        if not isinstance(tasks, list):
            tasks = [tasks]
        for task in tasks:
            self.__add_distinct_value_to_dict(self.tasks, task.name, task)

    def add_models(self, models):
        """
        Adds models to the manager. Models should be subclasses of BaseModel.
        """
        if not isinstance(models, list):
            models = [models]
        for model in models:
            self.__add_distinct_value_to_dict(self.models, model.name, model)

    def __extract_img_paths(self, df):
        try:
            return set(df['img1']).union(df['img2'])
        except Exception as e:
            raise e

    def __open_image(self, model, image_path):
        if hasattr(model, 'preprocess_image'):
            return model.preprocess_image(image_path)
        else:
            image = Image.open(image_path).convert('RGB')
            return model.preprocess(image).unsqueeze(0)

    def __get_pairs_output(self, model, pairs_df, images_folder_path):
        images_paths = self.__extract_img_paths(pairs_df)
        for img in images_paths:
            if img not in self.images_tensors[model.name]:
                image_tensor = self.__open_image(model, os.path.join(images_folder_path, img))
                self.images_tensors[model.name][img] = model.get_output(image_tensor)
        return self.images_tensors[model.name]

    def __compute_distances(self, model_name, distance_metric, df):
        distances = []
        for _, row in df.iterrows():
            img1_name = row['img1']
            img2_name = row['img2']
            d = distance_metric(self.images_tensors[model_name][img1_name], self.images_tensors[model_name][img2_name])
            distances.append(d)
        return distances

    def save_task_tensors_to_drive(self, model_name, task_name, path):
        if task_name not in self.tasks:
            raise Exception("Task Not Found!")
        torch.save(self.images_tensors[model_name], os.path.join(path, f"{date.today()}_{model_name}_{task_name}_tensors.pth"))
        print(f"Saved {task_name} tensors for {model_name} to drive.")

    def save_task_performance_df_to_drive(self, task_name, path):
        if task_name not in self.tasks:
            raise Exception("Task Not Found!")
        self.tasks_performance_dfs[task_name].to_csv(os.path.join(path, f"{date.today()}_{task_name}_performance.csv"))
        print(f"Saved {task_name} performance to drive.")

    def compute_tensors(self, model_name, task_name, print_log=False):
        if task_name not in self.tasks:
            raise Exception("Task Not Found!")
        if model_name not in self.models:
            raise Exception("Model Not Found!")

        selected_task = self.tasks[task_name]
        selected_model = self.models[model_name]
        self.__get_pairs_output(selected_model, selected_task.pairs_df, selected_task.images_path)
        if print_log:
            print(f"Processed all images for {task_name}")

    def run_task(self, model_name, task_name):
        self.compute_tensors(model_name, task_name)
        selected_task = self.tasks[task_name]

        # Compute distances
        pairs_df = selected_task.pairs_df
        distances = self.__compute_distances(model_name, selected_task.distance_metric, pairs_df)
        self.model_task_distances_dfs[model_name][task_name] = pairs_df.assign(model_computed_distance=distances)

        # Compute task's metric
        task_performance = selected_task.compute_task_performance(distances)
        task_result = pd.concat([pd.Series([model_name], name="Model Name"), task_performance], axis=1)

        # Update task performance dataframe
        if self.tasks_performance_dfs.get(task_name) is None:
            self.tasks_performance_dfs[task_name] = task_result
        else:
            self.tasks_performance_dfs[task_name] = pd.concat([self.tasks_performance_dfs[task_name], task_result], ignore_index=True)

    def run_all_tasks_with_model(self, model_name):
        for task_name in self.tasks:
            self.run_task(model_name, task_name)
        print(f"Processed all tasks for {model_name}")
        self.plot_superplot(model_name)

    def run_all_tasks_all_models(self):
        for model_name in self.models:
            self.run_all_tasks_with_model(model_name)
        print(f"Processed all tasks for all models")
        self.plot_superplot()

    def plot_superplot(self, model_name=None):
        accuracy_tasks = [task_name for task_name, task in self.tasks.items() if isinstance(task, AccuracyTask)]
        correlation_tasks = [task_name for task_name, task in self.tasks.items() if isinstance(task, CorrelationTask)]

        if model_name:
            self.plot_accuracy_tasks(accuracy_tasks, model_name)
            self.plot_correlation_tasks(correlation_tasks, model_name)
        else:
            for model in self.models:
                self.plot_accuracy_tasks(accuracy_tasks, model)
                self.plot_correlation_tasks(correlation_tasks, model)

    def plot_accuracy_tasks(self, task_names, model_name):
        accuracy_data = []

        for task_name in task_names:
            if task_name not in self.tasks_performance_dfs:
                print(f"Task {task_name} Not Found! Will ignore it")
                continue
            
            task_df = self.tasks_performance_dfs[task_name]
            if 'Accuracy' not in task_df.columns:
                raise Exception(f"Task {task_name} is not an Accuracy Task")

            task_df['Task'] = task_name
            task_df = task_df[task_df['Model Name'] == model_name]
            accuracy_data.append(task_df)

        combined_df = pd.concat(accuracy_data)

        plt.figure(figsize=(12, 8))
        sns.barplot(x="Task", y="Accuracy", hue="Model Name", data=combined_df)
        plt.title(f'Accuracy Comparison Across Multiple Tasks for {model_name}')
        plt.ylabel('Accuracy')
        plt.xlabel('Task')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def plot_correlation_tasks(self, task_names, model_name):
        for task_name in task_names:
            if task_name not in self.tasks_performance_dfs:
                raise Exception(f"Task {task_name} Not Found!")
            task_df = self.tasks_performance_dfs[task_name]
            if 'Correlation Score' not in task_df.columns:
                raise Exception(f"Task {task_name} is not a Correlation Task")

            model_distances = self.model_task_distances_dfs[model_name][task_name]
            
            plt.figure(figsize=(10, 6))
            plt.scatter(model_distances['img1'], model_distances['img2'], label=model_name)
            plt.title(f'Correlation Scatter Plot for Task: {task_name}')
            plt.xlabel('Image 1')
            plt.ylabel('Image 2')
            plt.legend()
            plt.show()