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
        image_tensor = self.model.preprocess_image(full_image_path)
        return image_tensor, image_path

class MultiModelTaskManager():
    """
    A helper class to work with multiple neural network models and tasks.

    Attributes:
        models: A dictionary of neural network models.
        tasks: Dictionary storing tasks information.
        model_task_distances_dfs: Dictionary storing distances computed between images, by task and model.
        images_tensors: Dictionary storing image tensors, by model.
        tasks_performance_dfs: Dictionary storing task results as dataframes, by task name.
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
        return (f'MultiModelTaskManager(models={list(self.models.keys())}, '
                f'tasks={list(self.tasks.keys())}')

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
                batch_outputs = model.get_output(image_tensors)

                for idx, img_name in enumerate(image_names):
                    outputs = {layer_name: output[idx] for layer_name, output in batch_outputs.items()}
                    self.images_tensors[model.name][img_name] = outputs

                # Clear variables to free memory
                del image_tensors, image_names, batch_outputs
                torch.cuda.empty_cache()
        return self.images_tensors[model.name]

    def __compute_distances(self, model_name, distance_metric, df):
        records = []
        sample_img_name = next(iter(self.images_tensors[model_name]))
        layer_names = self.images_tensors[model_name][sample_img_name].keys()

        for _, row in df.iterrows():
            img1_name = row['img1']
            img2_name = row['img2']
            pair_id = row['pair_id']

            for layer_name in layer_names:
                tensor1 = self.images_tensors[model_name][img1_name][layer_name]
                tensor2 = self.images_tensors[model_name][img2_name][layer_name]
                
                if tensor1.ndim == 1:
                    tensor1 = tensor1.unsqueeze(0)
                if tensor2.ndim == 1:
                    tensor2 = tensor2.unsqueeze(0)

                d = distance_metric(tensor1, tensor2)
                if isinstance(d, torch.Tensor):
                    d = d.item()

                record = {
                    'pair_id': pair_id,
                    'img1': img1_name,
                    'img2': img2_name,
                    'Layer Name': layer_name,
                    'nn_computed_distance': d
                }
                records.append(record)

        distances_df = pd.DataFrame.from_records(records)
        return distances_df


    def group_tasks_by_type(self):
        task_type_groups = defaultdict(list)

        for task_name, task_info in self.tasks.items():
            task_type = type(task_info)
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
            raise Exception('Task Not Found!')
        
        model_tensors = self.images_tensors[model_name]
        for layer_name, layer_outputs in model_tensors.items():
            save_path = os.path.join(path, f'{date.today()}_{model_name}_{task_name}_{layer_name}_tensors.pth')
            torch.save(layer_outputs, save_path)
            print(f'Saved {task_name} tensors for {model_name}, layer {layer_name}.')

    def export_model_task_performance(self, task_name, path):
        if task_name not in self.tasks:
            raise Exception('Task Not Found!')
        self.tasks_performance_dfs[task_name].to_csv(os.path.join(path, f'{date.today()}_{task_name}_performance.csv'))
        print(f'Saved {task_name} performance dataframe.')

    def export_model_results_by_task(self, export_path):
        res_by_task_type = self.group_tasks_by_type()
        for task_class, res in res_by_task_type.items():
            task_class_folder = os.path.join(export_path, task_class.__name__)
            os.makedirs(task_class_folder, exist_ok=True)

            # Exports results by task type
            res.to_csv(os.path.join(export_path, f'{date.today()}_all_{task_class.__name__}_results.csv'))
            # Uses the class's static plotting method
            task_class.plot(task_class_folder, res, self.model_task_distances_dfs)

    def compute_tensors(self, model_name, task_name, print_log=False):
        if task_name not in self.tasks:
            raise Exception('Task Not Found!')
        if model_name not in self.models:
            raise Exception('Model Not Found!')

        selected_task = self.tasks[task_name]
        selected_model = self.models[model_name]
        self.__get_pairs_output(selected_model, selected_task.pairs_df, selected_task.images_path)
        if print_log:
            print(f'Processed all images for {task_name}.')

    def run_task(self, model_name, task_name, export_path):
        self.compute_tensors(model_name, task_name)
        selected_task = self.tasks[task_name]

        pairs_df = selected_task.pairs_df
        pairs_df = pairs_df.assign(pair_id=pairs_df.index)
        distances_df = self.__compute_distances(model_name, selected_task.distance_metric, pairs_df)
        inputs_result_merge = pairs_df.merge(distances_df, on='pair_id', how='left')

        self.model_task_distances_dfs[model_name][task_name] = inputs_result_merge

        task_performance_list = []

        for layer_name, group_df in inputs_result_merge.groupby('Layer Name'):
            task_performance = selected_task.compute_task_performance(group_df)
            task_result = pd.DataFrame({
                'Model Name': [model_name],
                'Task Name': [task_name],
                'Layer Name': [layer_name],
                **task_performance.to_dict(orient='list')
            })
            task_performance_list.append(task_result)

        task_performance_df = pd.concat(task_performance_list, ignore_index=True)

        if task_name not in self.tasks_performance_dfs:
            self.tasks_performance_dfs[task_name] = task_performance_df
        else:
            self.tasks_performance_dfs[task_name] = pd.concat(
                [self.tasks_performance_dfs[task_name], task_performance_df],
                ignore_index=True
            )

    def run_task_with_all_models(self, task_name, export_path=os.getcwd()):
        for model_name in self.models:
            self.run_task(model_name, task_name, export_path)
        self.export_model_results_by_task(export_path)

    def run_all_tasks_with_model(self, model_name, export_path=os.getcwd()):
        for task_name in self.tasks:
            self.run_task(model_name, task_name, export_path)
        print(f'Processed all tasks for {model_name}.')

    def run_all_tasks_all_models(self, export_path=os.getcwd()):
        os.makedirs(export_path, exist_ok=True)
        for model_name in self.models:
            self.run_all_tasks_with_model(model_name)
        print(f'Finished processing all the tasks for all the models.')
        self.export_model_results_by_task(export_path)