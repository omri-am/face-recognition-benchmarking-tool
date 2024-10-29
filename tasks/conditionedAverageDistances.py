from facesBenchmarkUtils.baseTask import *
from typing import Any, Callable

class ConditionedAverageDistances(BaseTask):
    def __init__(
        self,
        name: str,
        pairs_file_path: str,
        images_path: str,
        distance_metric: Callable[[Any, Any], float],
        condition_column: str = 'condition',
        normalize: bool = True
    ) -> None:
        """
        Initializes the ConditionedAverageDistances instance.

        Parameters
        ----------
        name : str
            The name of the task.
        pairs_file_path : str
            Path to the CSV file containing image pairs and condition labels.
        images_path : str
            Path to the directory containing images.
        distance_metric : callable
            Function to compute the distance between image embeddings.
        condition_column : str
            Column name in the pairs file distinguishing between the different conditions. Default is 'condition'.
        normialize: bool
            Boolean parameter for normializing the computed distances, by deviding each distance with the max distance computed. Default is True.
        """
        super().__init__(
            name=name,
            pairs_file_path=pairs_file_path,
            images_path=images_path,
            distance_metric=distance_metric
        )
        self.distance_metric_name: str = distance_metric.__name__
        self.normalize: bool = normalize
        self.pairs_df.rename(columns = {condition_column:'condition'}, inplace=True)

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
        if self.normalize:
            max_distance = pairs_distances_df['model_computed_distance'].max()
            if max_distance != 0:
                pairs_distances_df['normalized_distance'] = pairs_distances_df['model_computed_distance'] / max_distance
        else:
            pairs_distances_df['normalized_distance'] = pairs_distances_df['model_computed_distance']

        avg_distances = pairs_distances_df.groupby(['condition'])['normalized_distance'].mean().reset_index()
        
        avg_distances.rename(columns={'normalized_distance': 'Mean Value', 'condition': 'Condition'}, inplace=True)
        
        avg_distances['Distance Metric'] = self.distance_metric_name

        return avg_distances
    
    @staticmethod
    def plot(
        output_dir: str,
        performances: pd.DataFrame,
        *optional: Any
    ) -> None:
        """
        Generates and saves a bar plot for each condition.

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
        for condition, condition_df in performances.groupby('Condition'):
            PlotHelper.bar_plot(
                performances=condition_df,
                y=f'Mean Value',
                ylabel='Average Distance',
                title_prefix=f'Average Distance Comparison - {condition}',
                output_dir=output_dir,
                file_name=f'average_distance_comparison: {condition}'
            )