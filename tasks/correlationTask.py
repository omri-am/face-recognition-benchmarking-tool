from facesBenchmarkUtils.baseTask import *
from typing import Any, Callable, Dict

class CorrelationTask(BaseTask):
    """
    A task that evaluates the correlation between the model's computed distances and the true distances.

    Attributes
    ----------
    correlation_function : callable
        The correlation metric to be used for evaluating the task.
    distance_function_name : str
        Name of the distance metric used.
    correlation_function_name : str
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
        task_name: str,
        pairs_file_path: str,
        images_folder_path: str,
        distance_function: Callable[[Any, Any], float],
        correlation_function: Callable[[Any, Any], np.ndarray] = np.corrcoef
    ) -> None:
        """
        Initializes the CorrelationTask instance.

        Parameters
        ----------
        task_name : str
            The name of the task.
        pairs_file_path : str
            Path to the CSV file containing image pairs and true distances.
        images_folder_path : str
            Path to the directory containing images.
        distance_function : callable
            Function to compute the distance between image embeddings.
        correlation_function : callable, optional
            Function to compute the correlation between computed distances and true distances.
            Defaults to numpy's corrcoef function.
        """
        super().__init__(
            task_name=task_name,
            pairs_file_path=pairs_file_path,
            images_folder_path=images_folder_path,
            distance_function=distance_function
        )
        self.correlation_function = correlation_function
        self.correlation_function_name = correlation_function.__name__
        if 'distance' not in self.pairs_df.columns:
            raise Exception('The pairs file must contain a "distance" column.')

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
        computed_distances = pairs_distances_df['model_computed_distance'].values
        true_distances = self.pairs_df['distance'].values

        correlation_result = self.correlation_function(computed_distances, true_distances)

        if isinstance(correlation_result, np.ndarray):
            correlation = correlation_result[0, 1]
        elif isinstance(correlation_result, tuple):
            correlation = correlation_result[0]
        else:
            correlation = correlation_result

        return pd.DataFrame({
            'Correlation Score': [correlation],
            'Distance Metric': [self.distance_function_name],
            'Correlation Metric':[self.correlation_function_name]
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
            ylim=(None, 1.1),
            title_prefix='Correlation Coefficient Comparison',
            output_dir=output_dir,
            file_name='correlation_comparison'
        )
