import os
import torch
import pandas as pd
from datetime import date
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
from .implementedTasks import *
from .utils import *

class PairDataset(Dataset):
    """
    A custom Dataset class that provides pairs of images for processing.

    Args:
        pairs_df (pd.DataFrame): DataFrame containing image pairs and their identifiers.
        images_folder_path (str): Path to the folder containing images.
        model: The model instance which includes the image preprocessing method.

    """

    def __init__(self, pairs_df, images_folder_path, model):
        self.pairs_df = pairs_df
        self.images_folder_path = images_folder_path
        self.model = model

    def __len__(self):
        """Returns the total number of image pairs."""
        return len(self.pairs_df)

    def __getitem__(self, idx):
        """
        Retrieves the image tensors and associated metadata for the given index.

        Args:
            idx (int): Index of the image pair to retrieve.

        Returns:
            tuple: A tuple containing:
                - img1_tensor (torch.Tensor): Preprocessed tensor of the first image.
                - img2_tensor (torch.Tensor): Preprocessed tensor of the second image.
                - img1_name (str): Filename of the first image.
                - img2_name (str): Filename of the second image.
                - pair_id (int): Unique identifier of the image pair.
        """
        row = self.pairs_df.iloc[idx]
        img1_name = row['img1']
        img2_name = row['img2']
        pair_id = row['pair_id']
        img1_path = os.path.join(self.images_folder_path, img1_name)
        img2_path = os.path.join(self.images_folder_path, img2_name)
        img1_tensor = self.model.preprocess_image(img1_path)
        img2_tensor = self.model.preprocess_image(img2_path)
        return img1_tensor, img2_tensor, img1_name, img2_name, pair_id

class MultiModelTaskManager():
    """
    Manages multiple models and tasks, facilitating the computation of task performance across models.

    Attributes:
        tasks (dict): Dictionary of task instances, keyed by task name.
        models (dict): Dictionary of model instances, keyed by model name.
        model_task_distances_dfs (dict): Nested dictionary storing computed distances for each model and task.
        tasks_performance_dfs (dict): Dictionary storing performance DataFrames for each task.
        batch_size (int): Batch size for data loading.

    """

    def __init__(self, models, tasks, batch_size=32):
        """
        Initializes the MultiModelTaskManager with given models and tasks.

        Args:
            models (list or object): A list of model instances or a single model instance.
            tasks (list or object): A list of task instances or a single task instance.
            batch_size (int, optional): Batch size for data loading. Defaults to 32.
        """
        self.tasks = {}
        self.add_tasks(tasks)
        self.models = {}
        self.add_models(models)
        self.model_task_distances_dfs = {
            model: {task_name: task.pairs_df for task_name, task in self.tasks.items()} 
            for model in self.models
        }
        self.tasks_performance_dfs = {}
        self.batch_size = batch_size

    def __repr__(self):
        """
        Returns a string representation of the MultiModelTaskManager.

        Returns:
            str: String representation of the instance.
        """
        return (f'MultiModelTaskManager(models={list(self.models.keys())}, '
                f'tasks={list(self.tasks.keys())}')

    def add_tasks(self, tasks):
        """
        Adds tasks to the manager.

        Args:
            tasks (list or object): A list of task instances or a single task instance.
        """
        if not isinstance(tasks, list):
            tasks = [tasks]
        for task in tasks:
            self.tasks[task.name] = task

    def add_models(self, models):
        """
        Adds models to the manager.

        Args:
            models (list or object): A list of model instances or a single model instance.
        """
        if not isinstance(models, list):
            models = [models]
        for model in models:
            self.models[model.name] = model

    def group_tasks_by_type(self):
        """
        Groups tasks by their class type.

        Returns:
            dict: Dictionary where keys are task types and values are DataFrames of task results.
        """
        task_type_groups = defaultdict(list)
        for task_name, task_info in self.tasks.items():
            task_type = type(task_info)
            task_type_groups[task_type].append(task_name)
        task_type_dfs = {}
        for task_type, task_names in task_type_groups.items():
            df_list = []
        for task_type, task_names in task_type_groups.items():
            df_list = [self.tasks_performance_dfs[task_name] for task_name in task_names]
            if df_list:
                task_type_dfs[task_type] = pd.concat(df_list)
        return task_type_dfs
                
    def export_computed_metrics(self, export_path):
        """
        Exports computed metrics for each model and task to CSV files.

        Args:
            export_path (str): Path to the directory where the metrics will be exported.
        """
        computed_folder = os.path.join(export_path, 'Computed Metrics')
        os.makedirs(computed_folder, exist_ok=True)

        for model_name, model_res in self.model_task_distances_dfs.items():
            model_folder = os.path.join(computed_folder, model_name)
            os.makedirs(model_folder, exist_ok=True)

            for task_name, model_task_res in model_res.items():
                for layer_name in self.models[model_name].extract_layers:
                    if 'Layer Name' in model_task_res.columns:
                        res_df = model_task_res[model_task_res['Layer Name'] == layer_name]
                    else:
                        res_df = model_task_res

                    layer_folder = (
                        os.path.join(model_folder)
                        if layer_name == 'default'
                        else os.path.join(model_folder, layer_name)
                    )
                    os.makedirs(layer_folder, exist_ok=True)

                    output_file = os.path.join(layer_folder, f'{task_name}.csv')
                    res_df.to_csv(output_file, index=False)

    def export_model_results_by_task(self, export_path):
        """
        Exports the model results grouped by task type and generates plots.

        Args:
            export_path (str): Path to the directory where the results will be exported.
        """
        res_by_task_type = self.group_tasks_by_type()
        for task_class, res in res_by_task_type.items():
            task_class_folder = os.path.join(export_path, f'{task_class.__name__} Plots')
            os.makedirs(task_class_folder, exist_ok=True)

            res.to_csv(
                os.path.join(task_class_folder, f'{task_class.__name__} results.csv'), index=False
            )
            task_class.plot(task_class_folder, res, self.model_task_distances_dfs)

    def export_unified_summary(self, export_path=os.getcwd()):
        """
        Exports a unified summary CSV file where each row corresponds to a model and layer,
        and columns correspond to the tasks with their performance metrics.

        Args:
            export_path (str, optional): Path to the directory where the summary will be exported.
                Defaults to the current working directory.
        """
        performance_dfs = []
        for _, task_performance_df in self.tasks_performance_dfs.items():
            performance_dfs.append(task_performance_df)

        all_performance_df = pd.concat(performance_dfs, ignore_index=True)
        all_performance_df.to_csv(
            os.path.join(export_path, f'{date.today()}_all performance df.csv'), index=False
        )

        id_columns = ['Model Name', 'Layer Name', 'Task Name']
        functions_columns = ['Distance Metric', 'Correlation Metric']
        metric_columns = [
            col for col in all_performance_df.columns if col not in id_columns + functions_columns
        ]

        summary_df = all_performance_df.set_index(id_columns)[metric_columns]
        summary_df = summary_df.unstack('Task Name')
        summary_df.columns = [
            ' '.join(col).strip() if isinstance(col, tuple) else col for col in summary_df.columns.values
        ]
        summary_df.reset_index(inplace=True)

        output_file = os.path.join(export_path, f'{date.today()}_models unified summary.csv')
        summary_df.to_csv(output_file, index=False)

    def run_task(self, model_name, task_name, export_path=os.getcwd(), print_log=False):
        """
        Runs a specific task on a specific model.

        Args:
            model_name (str): Name of the model to run the task on.
            task_name (str): Name of the task to run.
            export_path (str, optional): Path to the directory where results will be exported.
                Defaults to the current working directory.
            print_log (bool, optional): Whether to print log messages. Defaults to False.

        Raises:
            Exception: If the task or model is not found in the manager.
        """
        if task_name not in self.tasks:
            raise Exception('Task Not Found!')
        if model_name not in self.models:
            raise Exception('Model Not Found!')

        selected_task = self.tasks[task_name]
        selected_model = self.models[model_name]
        pairs_df = selected_task.pairs_df.copy()
        pairs_df = pairs_df.assign(pair_id=pairs_df.index)
        images_folder_path = selected_task.images_path

        # Create PairDataset
        dataset = PairDataset(pairs_df, images_folder_path, selected_model)
        if len(dataset) == 0:
            if print_log:
                print(f'No data to process for task "{task_name}" with model "{model_name}".')
            return
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # Process data and compute distances
        distances_df = self._process_batches(dataloader, selected_model, selected_task)

        # Merge results with pairs_df
        inputs_result_merge = pairs_df.merge(distances_df, on='pair_id', how='left')
        self.model_task_distances_dfs[model_name][task_name] = inputs_result_merge

        # Compute task performance
        task_performance_df = self._compute_task_performance(
            inputs_result_merge, selected_task, model_name, task_name
        )
        self._update_tasks_performance(task_name, task_performance_df)

        self.export_computed_metrics(export_path)
        self.export_unified_summary(export_path)

        if print_log:
            print(f'Processed task "{task_name}" for model "{model_name}".')

    def _process_batches(self, dataloader, model, task):
        """
        Processes batches of data from the DataLoader and computes distances.

        Args:
            dataloader (DataLoader): DataLoader for the PairDataset.
            model: The model instance used for feature extraction.
            task: The task instance containing the distance metric.

        Returns:
            pd.DataFrame: DataFrame containing computed distances for each pair.
        """
        records = []
        with torch.no_grad():
            for batch in dataloader:
                img1_tensors, img2_tensors, img1_names, img2_names, pair_ids = batch
                img1_tensors = img1_tensors.to(model.device)
                img2_tensors = img2_tensors.to(model.device)
                batch_outputs1 = model.get_output(img1_tensors)
                batch_outputs2 = model.get_output(img2_tensors)
                layer_names = batch_outputs1.keys()

                batch_records = self._compute_distances_for_batch(
                    batch_outputs1, batch_outputs2, img1_names, img2_names, pair_ids, layer_names, task
                )
                records.extend(batch_records)

                # Clear variables to free memory
                del img1_tensors, img2_tensors, batch_outputs1, batch_outputs2
                torch.cuda.empty_cache()
        distances_df = pd.DataFrame.from_records(records)
        return distances_df

    def _compute_distances_for_batch(
        self, outputs1, outputs2, img1_names, img2_names, pair_ids, layer_names, task
    ):
        """
        Computes distances for a batch of outputs.

        Args:
            outputs1 (dict): Dictionary of layer outputs for the first set of images.
            outputs2 (dict): Dictionary of layer outputs for the second set of images.
            img1_names (list): List of image names for the first set.
            img2_names (list): List of image names for the second set.
            pair_ids (list): List of pair identifiers.
            layer_names (list): List of layer names.
            task: The task instance containing the distance metric.

        Returns:
            list: List of records containing computed distances and metadata.
        """
        records = []
        for i in range(len(pair_ids)):
            pair_id = pair_ids[i].item()
            img1_name = img1_names[i]
            img2_name = img2_names[i]
            for layer_name in layer_names:
                tensor1 = outputs1[layer_name][i]
                tensor2 = outputs2[layer_name][i]

                # Adjust dimensions if necessary
                if tensor1.ndim == 1:
                    tensor1 = tensor1.unsqueeze(0)
                if tensor2.ndim == 1:
                    tensor2 = tensor2.unsqueeze(0)

                d = task.distance_metric(tensor1, tensor2)
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
        return records

    def _compute_task_performance(self, inputs_result_merge, task, model_name, task_name):
        """
        Computes the performance metrics for a task.

        Args:
            inputs_result_merge (pd.DataFrame): DataFrame containing input data and computed distances.
            task: The task instance containing the performance computation method.
            model_name (str): Name of the model.
            task_name (str): Name of the task.

        Returns:
            pd.DataFrame: DataFrame containing performance metrics.
        """
        task_performance_list = []
        for layer_name, group_df in inputs_result_merge.groupby('Layer Name'):
            task_performance = task.compute_task_performance(group_df)
            task_result = pd.DataFrame({
                'Task Name': [task_name],
                'Model Name': [model_name],
                'Layer Name': [layer_name],
                **task_performance.to_dict(orient='list')
            })
            task_performance_list.append(task_result)
        task_performance_df = pd.concat(task_performance_list, ignore_index=True)
        return task_performance_df

    def _update_tasks_performance(self, task_name, task_performance_df):
        """
        Updates the tasks_performance_dfs dictionary with new performance data.

        Args:
            task_name (str): Name of the task.
            task_performance_df (pd.DataFrame): DataFrame containing performance metrics.
        """
        if task_name not in self.tasks_performance_dfs:
            self.tasks_performance_dfs[task_name] = task_performance_df
        else:
            self.tasks_performance_dfs[task_name] = pd.concat(
                [self.tasks_performance_dfs[task_name], task_performance_df],
                ignore_index=True
            )
        self.tasks_performance_dfs[task_name] = sort_mixed_column(
            df=self.tasks_performance_dfs[task_name], 
            column_name='Layer Name', 
            additional_sort_fields=['Task Name', 'Model Name']
        )

    def run_task_with_all_models(self, task_name, export_path=os.getcwd(), print_log=False):
        """
        Runs a specific task on all models and exports the results.

        Args:
            task_name (str): Name of the task to run.
            export_path (str, optional): Path to the directory where results will be exported.
                Defaults to the current working directory.
            print_log (bool, optional): Whether to print log messages. Defaults to False.
        """
        for model_name in self.models:
            self.run_task(model_name, task_name, export_path, print_log)
        self.export_model_results_by_task(export_path)

    def run_all_tasks_with_model(self, model_name, export_path=os.getcwd(), print_log=False):
        """
        Runs all tasks on a specific model and exports the results.

        Args:
            model_name (str): Name of the model to run tasks on.
            export_path (str, optional): Path to the directory where results will be exported.
                Defaults to the current working directory.
            print_log (bool, optional): Whether to print log messages. Defaults to False.
        """
        for task_name in self.tasks:
            self.run_task(model_name, task_name, export_path, print_log)
        if print_log:
            print(f'Processed all tasks for {model_name}.')

    def run_all_tasks_all_models(self, export_path=os.getcwd(), print_log=False):
        """
        Runs all tasks on all models and exports the results.

        Args:
            export_path (str, optional): Path to the directory where results will be exported.
                Defaults to the current working directory.
            print_log (bool, optional): Whether to print log messages. Defaults to False.
        """
        os.makedirs(export_path, exist_ok=True)
        for model_name in self.models:
            self.run_all_tasks_with_model(model_name, export_path, print_log)
        if print_log:
            print(f'Finished processing all the tasks for all the models.')
        self.export_model_results_by_task(export_path)
        if print_log:
            print(f'Finished plotting all the tasks for all the models.')