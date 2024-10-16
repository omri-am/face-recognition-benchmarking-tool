import os
import torch
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise
import matplotlib.pyplot as plt
import seaborn as sns
import math
from .utils import *

SUPTITLE_SIZE = 18
SUBTITLE_SIZE = 12

class PlotHelper():
    def bar_plot(performances, y, ylabel, title_prefix, output_dir, file_name, ylim=None):
        for task_name, task_df in performances.groupby('Task Name'):
            task_data = task_df.copy()
            task_data['Model-Layer'] = task_data['Model Name'] + ': ' + task_data['Layer Name']
            task_data = sort_mixed_column(task_data, 'Model-Layer')

            width = max(round(len(task_data['Model-Layer'].unique())/20 * 12), 12)
            plt.figure(figsize=(width, 8))
            
            ax = sns.barplot(x='Model-Layer', y=y, data=task_data, hue='Model-Layer')
            
            locations = ax.get_xticks()
            labels = [item.get_text() for item in ax.get_xticklabels()]
            
            ax.set_xticks(locations)
            ax.set_xticklabels(labels, rotation=45, ha='right')
            
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.2f}',
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=11, color='black', 
                            xytext=(0, 5), textcoords='offset points')

            plt.title(f'{title_prefix}: {task_name}', fontsize=SUPTITLE_SIZE)
            plt.ylabel(ylabel)
            plt.xlabel('Model-Layer')
            if ylim:
                plt.ylim(ylim)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{task_name}_{file_name}.png'), bbox_inches='tight')
            plt.close()

    def scatter_plot(performances, distances, output_dir, file_name):
        for task_name, task_df in performances.groupby('Task Name'):
            task_data = task_df.copy()
            task_data['Model-Layer'] = task_data['Model Name'] + ': ' + task_data['Layer Name']
            task_data = sort_mixed_column(task_data, 'Model-Layer')

            num_plots = len(task_data)
            cols = math.ceil(math.sqrt(num_plots))
            rows = math.ceil(num_plots / cols)

            min_y = min(distances[model_name][task_name]['nn_computed_distance'].min() for model_name in task_df['Model Name'].unique())
            max_y = max(distances[model_name][task_name]['nn_computed_distance'].max() for model_name in task_df['Model Name'].unique())

            fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
            axs = axs.flatten() if isinstance(axs, np.ndarray) else np.array([axs])

            idx = 0
            for _, row in task_data.iterrows():
                model_name = row['Model Name']
                layer_name = row['Layer Name']
                correlation_score = row['Correlation Score']
                
                d_data = distances[model_name][task_name]
                d_data_layer = d_data[d_data['Layer Name'] == layer_name]
                
                ax = axs[idx]
                sns.regplot(data=d_data_layer,
                            x='distance',
                            y='nn_computed_distance',
                            scatter=True,
                            line_kws={'color': 'red'},
                            ax=ax)
                ax.set_title(f'{model_name} - {layer_name}\nCorrelation: {correlation_score:.2f}',
                            fontsize=SUBTITLE_SIZE)
                ax.set_xlabel('Input File Distance')
                ax.set_ylabel('NN Computed Distance')
                ax.set_ylim(max(min_y-0.05, 0), min(max_y+0.05, 1))
                idx += 1
            
            for i in range(idx, len(axs)):
                fig.delaxes(axs[i])
            
            plt.suptitle(f'Correlation Scatter Plots: {task_name}', fontsize=SUPTITLE_SIZE)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(output_dir, f'{task_name}_{file_name}.png'))
            plt.close()

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
        if isinstance(x, torch.Tensor):
            if x.numel() == 1:
                return x.item()
            else:
                return x.mean().item()
        elif isinstance(x, np.ndarray):
            if x.size == 1:
                return float(x.item())
            else:
                return float(x.mean())
        elif np.isscalar(x):
            return float(x)
        else:
            return float(x)


    def __validate_distance_metric(self, user_func):
        try:
            rand_t1 = torch.rand((10,2))
            rand_t2 = torch.rand((10,2))
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
            raise FileNotFoundError(f"File Not Found! Please provide the full path of the file.\nPath provided: {path}")
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
    def compute_task_performance(self, pairs_distances_df):
        pass
