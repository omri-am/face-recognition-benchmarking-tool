from .baseTask import *
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

class AccuracyTask(BaseTask):
    """
    A task that evaluates the accuracy of the model's predictions.

    Attributes
    ----------
    true_label : str
        Column name in the pairs DataFrame indicating the ground truth labels.
    distance_metric_name : str
        Name of the distance metric used.

    Methods
    -------
    compute_task_performance(pairs_distances_df: pd.DataFrame) -> pd.DataFrame
        Computes the accuracy, AUC, and optimal threshold for the task.
    plot(output_dir: str, performances: pd.DataFrame, *optional: Any) -> None
        Generates and saves a bar plot of accuracy scores.
    """

    def __init__(
        self,
        name: str,
        pairs_file_path: str,
        images_path: str,
        distance_metric: Callable[[Any, Any], float],
        true_label: str = 'truth'
    ) -> None:
        """
        Initializes the AccuracyTask instance.

        Parameters
        ----------
        name : str
            The name of the task.
        pairs_file_path : str
            Path to the CSV file containing image pairs and labels.
        images_path : str
            Path to the directory containing images.
        distance_metric : callable
            Function to compute the distance between image embeddings.
        true_label : str, optional
            Column name in the pairs file indicating the ground truth labels.
            Defaults to 'truth'.
        """
        super().__init__(
            name=name,
            pairs_file_path=pairs_file_path,
            images_path=images_path,
            distance_metric=distance_metric
        )
        self.true_label: str = true_label
        self.distance_metric_name: str = distance_metric.__name__

    def compute_task_performance(self, pairs_distances_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the accuracy, AUC, and optimal threshold for the task.

        Parameters
        ----------
        pairs_distances_df : pandas.DataFrame
            DataFrame containing the computed distances for image pairs.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the accuracy, optimal threshold, AUC, and distance metric name.
        """
        pairs_distances_df['similarity'] = 1 - pairs_distances_df['nn_computed_distance']
        y_true = self.pairs_df[self.true_label].values
        y_scores = pairs_distances_df['similarity'].values

        auc = roc_auc_score(y_true, y_scores)

        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        y_pred = (y_scores > optimal_threshold).astype(int)
        accuracy = accuracy_score(y_true, y_pred)

        return pd.DataFrame({
            'Accuracy': [round(accuracy, 5)],
            'Optimal Threshold': [round(optimal_threshold, 5)],
            'AUC': [round(auc, 5)],
            'Distance Metric': [self.distance_metric_name]
        })

    @staticmethod
    def plot(
        output_dir: str,
        performances: pd.DataFrame,
        *optional: Any
    ) -> None:
        """
        Generates and saves a bar plot of accuracy scores.

        Parameters
        ----------
        output_dir : str
            Directory where the plot image will be saved.
        performances : pandas.DataFrame
            DataFrame containing performance metrics for each model and task.
        *optional : Any
            Additional optional arguments (not used).

        Returns
        -------
        None
        """
        PlotHelper.bar_plot(
            performances=performances,
            y='Accuracy',
            ylabel='Accuracy',
            ylim=(0, 1.1),
            title_prefix='Accuracy Score Comparison',
            output_dir=output_dir,
            file_name='accuracy_comparison'
        )

class CorrelationTask(BaseTask):
    """
    A task that evaluates the correlation between the model's computed distances and the true distances.

    Attributes
    ----------
    correlation_metric : callable
        The correlation metric to be used for evaluating the task.
    distance_metric_name : str
        Name of the distance metric used.
    correlation_metric_name : str
        Name of the correlation metric used.

    Methods
    -------
    compute_task_performance(pairs_distances_df: pd.DataFrame) -> pd.DataFrame
        Computes the correlation score for the task.
    plot(output_dir: str, performances: pd.DataFrame, distances: Dict[str, Dict[str, pd.DataFrame]], *optional: Any) -> None
        Generates and saves scatter plots showing the correlation.
    """

    def __init__(
        self,
        name: str,
        pairs_file_path: str,
        images_path: str,
        distance_metric: Callable[[Any, Any], float],
        correlation_metric: Callable[[Any, Any], np.ndarray] = np.corrcoef
    ) -> None:
        """
        Initializes the CorrelationTask instance.

        Parameters
        ----------
        name : str
            The name of the task.
        pairs_file_path : str
            Path to the CSV file containing image pairs and true distances.
        images_path : str
            Path to the directory containing images.
        distance_metric : callable
            Function to compute the distance between image embeddings.
        correlation_metric : callable, optional
            Function to compute the correlation between computed distances and true distances.
            Defaults to numpy's corrcoef function.
        """
        super().__init__(
            name=name,
            pairs_file_path=pairs_file_path,
            images_path=images_path,
            distance_metric=distance_metric
        )
        self.correlation_metric: Callable[[Any, Any], np.ndarray] = correlation_metric
        self.distance_metric_name: str = distance_metric.__name__
        self.correlation_metric_name: str = correlation_metric.__name__

    def compute_task_performance(self, pairs_distances_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the correlation score for the task.

        Parameters
        ----------
        pairs_distances_df : pandas.DataFrame
            DataFrame containing the computed distances for image pairs.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the correlation score, distance metric name, and correlation metric name.
        """
        computed_distances = pairs_distances_df['nn_computed_distance'].values
        true_distances = self.pairs_df['distance'].values

        correlation_result = self.correlation_metric(computed_distances, true_distances)

        if isinstance(correlation_result, np.ndarray):
            correlation = correlation_result[0, 1]
        elif isinstance(correlation_result, tuple):
            correlation = correlation_result[0]
        else:
            correlation = correlation_result

        return pd.DataFrame({
            'Correlation Score': [round(correlation, 5)],
            'Distance Metric': [self.distance_metric_name],
            'Correlation Metric':[self.correlation_metric_name]
        })
    
    @staticmethod
    def plot(
        output_dir: str,
        performances: pd.DataFrame,
        distances: Dict[str, Dict[str, pd.DataFrame]],
        *optional: Any
    ) -> None:
        """
        Generates and saves scatter plots showing the correlation, and bar plot comparing R values.

        Parameters
        ----------
        output_dir : str
            Directory where the plot images will be saved.
        performances : pandas.DataFrame
            DataFrame containing performance metrics for each model and task.
        distances : dict
            Nested dictionary containing distance DataFrames for each model and task.
        *optional : Any
            Additional optional arguments (not used).

        Returns
        -------
        None
        """
        PlotHelper.scatter_plot(
            performances=performances,
            distances=distances,
            output_dir=output_dir,
            file_name='scatters_comparison'
        )
        PlotHelper.bar_plot(
            performances=performances,
            y='Correlation Score',
            ylabel='Correlation Coefficient (r)',
            ylim=(0, 1.1),
            title_prefix='Correlation Coefficient Comparison',
            output_dir=output_dir,
            file_name='correlation_comparison'
        )
    
class RelativeDifferenceTask(BaseTask):
    """
    A task that computes the relative difference between two groups in the computed distances.

    Attributes
    ----------
    group_column : str
        Column name in the pairs DataFrame distinguishing between the two groups.
    distance_metric_name : str
        Name of the distance metric used.

    Methods
    -------
    compute_task_performance(pairs_distances_df: pd.DataFrame) -> pd.DataFrame
        Computes the relative difference between two groups.
    plot(output_dir: str, performances: pd.DataFrame, *optional: Any) -> None
        Generates and saves a bar plot of relative differences.
    """

    def __init__(
        self,
        name: str,
        pairs_file_path: str,
        images_path: str,
        distance_metric: Callable[[Any, Any], float],
        group_column: str
    ) -> None:
        """
        Initializes the RelativeDifferenceTask instance.

        Parameters
        ----------
        name : str
            The name of the task.
        pairs_file_path : str
            Path to the CSV file containing image pairs and group labels.
        images_path : str
            Path to the directory containing images.
        distance_metric : callable
            Function to compute the distance between image embeddings.
        group_column : str
            Column name in the pairs file distinguishing between the two groups.
        """
        super().__init__(
            name=name,
            pairs_file_path=pairs_file_path,
            images_path=images_path,
            distance_metric=distance_metric
        )
        self.group_column: str = group_column
        self.distance_metric_name: str = distance_metric.__name__

    def compute_task_performance(self, pairs_distances_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes the relative difference between two groups.

        Parameters
        ----------
        pairs_distances_df : pandas.DataFrame
            DataFrame containing the computed distances for image pairs.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the group means, relative difference, and distance metric name.
        """
        unique_groups = pairs_distances_df[self.group_column].unique()
        if len(unique_groups) != 2:
            raise ValueError(
                f"The group column '{self.group_column}' must have exactly two unique values."
            )

        group1 = pairs_distances_df[pairs_distances_df[self.group_column] == unique_groups[0]]
        group2 = pairs_distances_df[pairs_distances_df[self.group_column] == unique_groups[1]]

        group1_mean = group1['nn_computed_distance'].mean()
        group2_mean = group2['nn_computed_distance'].mean()

        relative_difference = (group1_mean - group2_mean) / (group1_mean + group2_mean)

        return pd.DataFrame({
            'Group 1 Mean': [group1_mean],
            'Group 2 Mean': [group2_mean],
            'Relative Difference': [round(relative_difference, 5)],
            'Distance Metric': [self.distance_metric_name]
        })

    @staticmethod
    def plot(
        output_dir: str,
        performances: pd.DataFrame,
        *optional: Any
    ) -> None:
        """
        Generates and saves a bar plot of relative differences.

        Parameters
        ----------
        output_dir : str
            Directory where the plot images will be saved.
        performances : pandas.DataFrame
            DataFrame containing performance metrics for each model and task.
        *optional : Any
            Additional optional arguments (not used).

        Returns
        -------
        None
        """
        PlotHelper.bar_plot(
            performances=performances,
            y='Relative Difference',
            ylabel='Relative Difference',
            title_prefix='Relative Difference Comparison',
            output_dir=output_dir,
            file_name='relative_difference_comparison'
        )