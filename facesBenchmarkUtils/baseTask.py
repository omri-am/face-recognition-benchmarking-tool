import os
import torch
from abc import ABC, abstractmethod
from typing import Any, Callable, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import pairwise
import matplotlib.pyplot as plt
import seaborn as sns
import math

SUPTITLE_SIZE = 18
SUBTITLE_SIZE = 12

create_model_layer_column = lambda row: row['Model Name'] if row['Layer Name'] == 'default' else row['Model Name'] + ': ' + row['Layer Name']

def get_function_name(func):
    return getattr(func, '__name__', str(func))

class PlotHelper:
    """
    A helper class for generating plots related to model performance on tasks.
    """

    @staticmethod
    def bar_plot(performances, y, ylabel, title_prefix, output_dir, file_name, ylim=None):
        """
        Creates and saves bar plots for task performances.

        Parameters
        ----------
        performances : pandas.DataFrame
            DataFrame containing performance metrics for each model and task.
        y : str
            The column name in `performances` to be used as the y-axis value.
        ylabel : str
            Label for the y-axis.
        title_prefix : str
            Prefix for the plot titles.
        output_dir : str
            Directory where the plot images will be saved.
        file_name : str
            Base file name for the saved plot images.
        ylim : tuple of float, optional
            Y-axis limits for the plots as (min, max).

        Returns
        -------
        None
        """
        for task_name, task_df in performances.groupby('Task Name'):
            task_data = task_df.copy()
            task_data['Model-Layer'] = task_data.apply(create_model_layer_column, axis=1)

            width = max(round(len(task_data['Model-Layer'].unique()) / 20 * 12), 12)
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

    @staticmethod
    def scatter_plot(performances, distances, output_dir, file_name):
        """
        Creates and saves scatter plots showing the correlation between input distances and model-computed distances.

        Parameters
        ----------
        performances : pandas.DataFrame
            DataFrame containing performance metrics for each model and task.
        distances : dict
            Nested dictionary containing distance DataFrames for each model and task.
        output_dir : str
            Directory where the plot images will be saved.
        file_name : str
            Base file name for the saved plot images.

        Returns
        -------
        None
        """
        for task_name, task_df in performances.groupby('Task Name'):
            task_data = task_df.copy()
            task_data['Model-Layer'] = task_data.apply(create_model_layer_column, axis=1)

            num_plots = len(task_data)
            cols = math.ceil(math.sqrt(num_plots))
            rows = math.ceil(num_plots / cols)

            min_y = min(
                distances[model_name][task_name]['model_computed_distance'].min()
                for model_name in task_df['Model Name'].unique()
            )
            max_y = max(
                distances[model_name][task_name]['model_computed_distance'].max()
                for model_name in task_df['Model Name'].unique()
            )

            fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
            axs = axs.flatten() if isinstance(axs, np.ndarray) else np.array([axs])

            idx = 0
            for _, row in task_data.iterrows():
                model_name = row['Model Name']
                layer_name = row['Layer Name']
                correlation_score = row['Correlation Score']

                d_data = distances[model_name][task_name]
                d_data_layer = d_data[d_data['layer_name'] == layer_name]

                ax = axs[idx]
                sns.regplot(
                    data=d_data_layer,
                    x='distance',
                    y='model_computed_distance',
                    scatter=True,
                    line_kws={'color': 'red'},
                    ax=ax
                )
                ax.set_title(f'{model_name} - {layer_name}\nCorrelation: {correlation_score:.2f}',
                             fontsize=SUBTITLE_SIZE)
                ax.set_xlabel('Input File Distance')
                ax.set_ylabel('NN Computed Distance')
                ax.set_ylim(max(min_y - 0.05, 0), min(max_y + 0.05, 1))
                idx += 1

            for i in range(idx, len(axs)):
                fig.delaxes(axs[i])

            plt.suptitle(f'Correlation Scatter Plots: {task_name}', fontsize=SUPTITLE_SIZE)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(output_dir, f'{task_name}_{file_name}.png'))
            plt.close()

class BaseTask(ABC):
    """
    An abstract base class to represent a task in the benchmarking framework.
    All specific task classes should inherit from this class.

    Attributes
    ----------
    name : str
        The name of the task.
    images_path : str
        The directory where the images are stored.
    pairs_file_path : str
        The path to the CSV file containing image pairs.
    pairs_df : pandas.DataFrame
        DataFrame containing pairs of images and related information.
    distance_metric : callable
        The function used to compute the distance between image embeddings.
    """

    def __init__(
        self,
        name: str,
        pairs_file_path: str,
        images_path: str,
        distance_metric: Callable[[Any, Any], float] = pairwise.cosine_distances
    ) -> None:
        """
        Initializes the BaseTask instance.

        Parameters
        ----------
        name : str
            The name of the task.
        pairs_file_path : str
            Path to the CSV file containing image pairs.
        images_path : str
            Path to the directory containing images.
        distance_metric : callable, optional
            Function to compute the distance between image embeddings. Default is cosine distance.

        Raises
        ------
        FileNotFoundError
            If the provided image path or pairs file path does not exist.
        Exception
            If the pairs file does not contain required columns.
        """
        self.name = name
        self.pairs_file_path = pairs_file_path
        self.pairs_df = self.__load_file(pairs_file_path)
        self.images_path = self.__validate_path(images_path)
        self.distance_metric, self.distance_metric_name = self.__validate_and_set_distance_metric(distance_metric)

    def __to_float(self, x: Any) -> float:
        """
        Converts input to a float value. Supports torch.Tensor, numpy.ndarray, and scalars.

        Parameters
        ----------
        x : torch.Tensor or numpy.ndarray or scalar
            The input value to convert.

        Returns
        -------
        float
            The converted float value.
        """
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

    def __validate_and_set_distance_metric(
            self, 
            user_func: Callable[[Any, Any], Any]
            ) -> Tuple[Callable[[Any, Any], float], str]:
        """
        Validates the user-provided distance metric function.

        Parameters
        ----------
        user_func : callable
            The distance metric function to validate.

        Returns
        -------
        callable
            The validated distance metric function that returns a float.

        Raises
        ------
        Exception
            If the distance metric function is invalid.
        """
        try:
            rand_t1 = torch.rand((10, 2))
            rand_t2 = torch.rand((10, 2))
            result = user_func(rand_t1, rand_t2)
            self.__to_float(result)
        except Exception as e:
            raise Exception("Distance metric is not valid!") from e

        try:
            self.__to_float(user_func(rand_t1, rand_t2))
        except Exception:
            print(
                "WARNING! The distance function does not return a scalar or an array. "
                "This could potentially affect computing. Please consider changing the function."
            )
        return lambda x, y: self.__to_float(user_func(x, y)), user_func.__name__

    def __validate_path(self, path: str) -> str:
        """
        Validates that the provided path exists.

        Parameters
        ----------
        path : str
            The file or directory path to validate.

        Returns
        -------
        str
            The validated path.

        Raises
        ------
        FileNotFoundError
            If the path does not exist.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"File Not Found! Please provide the full path of the file.\nPath provided: {path}"
            )
        return path

    def __load_file(self, pairs_file_path: str) -> pd.DataFrame:
        """
        Loads the pairs CSV file into a DataFrame.

        Parameters
        ----------
        pairs_file_path : str
            The path to the pairs CSV file.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the image pairs.

        Raises
        ------
        Exception
            If the required columns are not present in the CSV file.
        """
        self.__validate_path(pairs_file_path)
        try:
            pairs_pd = pd.read_csv(pairs_file_path)
            if not {'img1', 'img2'}.issubset(pairs_pd.columns):
                raise Exception("The CSV file must contain 'img1' and 'img2' columns.")
            return pairs_pd
        except Exception as e:
            raise e

    @abstractmethod
    def compute_task_performance(self, pairs_distances_df: pd.DataFrame) -> pd.DataFrame:
        """
        Abstract method to compute the task performance metrics.

        Parameters
        ----------
        pairs_distances_df : pandas.DataFrame
            DataFrame containing the computed distances for image pairs.

        Returns
        -------
        pd.DataFrame
            The performance metrics DataFrame for the task.

        Raises
        ------
        NotImplementedError
            Must be implemented in subclasses.
        """
        pass

    @abstractmethod
    def plot(self,
        output_dir: str,
        performances: pd.DataFrame,
        *optional: Any):
        """
        --- domunetation here ---
        """
        pass