import os
import torch
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
from .implementedTasks import *

class ImageDataset(Dataset):
    def __init__(self, image_paths, images_folder_path, model):
        self.image_paths = list(image_paths)
        self.images_folder_path = images_folder_path
        self.model = model

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        full_image_path = os.path.join(self.images_folder_path, image_path)
        image_tensor = self.__open_image(self.model, full_image_path)
        return image_tensor.squeeze(0), image_path

    def __open_image(self, model, image_path):
        if hasattr(model, 'preprocess_image'):
            tensor = model.preprocess_image(image_path)
        else:
            image = Image.open(image_path).convert('RGB')
            tensor = model.preprocess(image)
        return tensor

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
    def __init__(self, models, tasks, batch_size=32):
        self.tasks = {}
        self.add_tasks(tasks)
        self.models = {}
        self.add_models(models)
        self.model_task_distances_dfs = {model: {task_name: task.pairs_df for task_name, task in self.tasks.items()} for model in self.models}
        self.images_tensors = {model: {} for model in self.models}
        self.tasks_performance_dfs = {task: None for task in self.tasks}
        self.batch_size = batch_size

    def __repr__(self):
        return (f"MultiModelTaskManager(models={list(self.models.keys())}, "
                f"tasks={list(self.tasks.keys())}")

    def add_tasks(self, tasks):
        """
        Adds tasks to the manager. Tasks should be subclasses of BaseTask.
        """
        if not isinstance(tasks, list):
            tasks = [tasks]
        for task in tasks:
            self.tasks[task.name] = task

    def add_models(self, models):
        """
        Adds models to the manager. Models should be subclasses of BaseModel.
        """
        if not isinstance(models, list):
            models = [models]
        for model in models:
            self.models[model.name] = model

    def __extract_img_paths(self, df):
        try:
            return set(df['img1']).union(df['img2'])
        except Exception as e:
            raise e

    def __get_pairs_output(self, model, pairs_df, images_folder_path):
        images_paths = self.__extract_img_paths(pairs_df)
        dataset = ImageDataset(images_paths, images_folder_path, model)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        with torch.no_grad():
            for image_tensors, image_names in dataloader:
                image_tensors = image_tensors.to(model.device)
                batch_outputs = []

                for img_tensor in image_tensors:
                    output = model.get_output(img_tensor)
                    batch_outputs.append(output.cpu().numpy())  

                for img, output in zip(image_names, batch_outputs):
                    self.images_tensors[model.name][img] = output
        return self.images_tensors[model.name]

    def __compute_distances(self, model_name, distance_metric, df):
        distances = []
        pair_ids = []

        for _, row in df.iterrows():
            img1_name = row['img1']
            img2_name = row['img2']
            pair_id = row['pair_id']

            tensor1 = self.images_tensors[model_name][img1_name]
            tensor2 = self.images_tensors[model_name][img2_name]
            
            # Reshape tensors if necessary
            if tensor1.ndim == 1:
                tensor1 = tensor1.reshape(1, -1)
            if tensor2.ndim == 1:
                tensor2 = tensor2.reshape(1, -1)

            d = distance_metric(tensor1, tensor2)
            distances.append(d)
            pair_ids.append(pair_id)

        return pd.DataFrame({'pair_id': pair_ids, 'nn_computed_distance': distances})

    def create_task_type_dataframes(self):
        task_type_groups = defaultdict(list)

        for task_name, task_info in self.tasks.items():
            task_type = type(task_info).__name__
            task_type_groups[task_type].append(task_name)

        task_type_dfs = {}

        for task_type, task_names in task_type_groups.items():
            df_list = []
            
            for task_name in task_names:
                if task_name in self.tasks_performance_dfs:
                    df_list.append(self.tasks_performance_dfs[task_name])

            if df_list:
                task_type_dfs[task_type] = pd.concat(df_list)

        return task_type_dfs

    def export_task_tensors(self, model_name, task_name, path):
        if task_name not in self.tasks:
            raise Exception("Task Not Found!")
        torch.save(self.images_tensors[model_name], os.path.join(path, f"{date.today()}_{model_name}_{task_name}_tensors.pth"))
        print(f"Saved {task_name} tensors for {model_name} dataframe.")

    def export_task_performance_df(self, task_name, path):
        if task_name not in self.tasks:
            raise Exception("Task Not Found!")
        self.tasks_performance_dfs[task_name].to_csv(os.path.join(path, f"{date.today()}_{task_name}_performance.csv"))
        print(f"Saved {task_name} performance dataframe.")

    def export_performances_by_task_type(self, path):
        res_by_task_type = self.create_task_type_dataframes()
        for task_type, res in res_by_task_type.items():
            res.to_csv(os.path.join(path, f"{date.today()}_all_{task_type}_results.csv"))

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
        pairs_df = pairs_df.assign(pair_id=pairs_df.index)
        distances_df = self.__compute_distances(model_name, selected_task.distance_metric, pairs_df)
        result_df = pairs_df.merge(distances_df, on='pair_id', how='left')

        self.model_task_distances_dfs[model_name][task_name] = result_df

        # Compute task's metric
        task_performance = selected_task.compute_task_performance(result_df)
        task_result = pd.concat([pd.Series([model_name], name="Model Name"), task_performance], axis=1)
        task_result['Task'] = task_name 

        # Update task performance dataframe
        if task_name not in self.tasks_performance_dfs:
            self.tasks_performance_dfs[task_name] = task_result
        else:
            self.tasks_performance_dfs[task_name] = pd.concat(
                [self.tasks_performance_dfs[task_name], task_result], 
                ignore_index=True
            )

    def run_all_tasks_with_model(self, model_name):
        for task_name in self.tasks:
            self.run_task(model_name, task_name)
        print(f"Processed all tasks for {model_name}")
        # self.plot_superplot(model_name)

    def run_all_tasks_all_models(self):
        for model_name in self.models:
            self.run_all_tasks_with_model(model_name)
        print(f"Processed all tasks for all models")
        self.export_performances_by_task_type(os.getcwd())
        # self.plot_superplot()

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