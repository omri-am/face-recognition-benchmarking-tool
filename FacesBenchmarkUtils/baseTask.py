import os
import torch
from torch import nn
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, pairwise

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
