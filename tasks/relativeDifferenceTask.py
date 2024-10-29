from facesBenchmarkUtils.baseTask import *
from typing import Any, Callable

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

        group1_mean = group1['model_computed_distance'].mean()
        group2_mean = group2['model_computed_distance'].mean()

        relative_difference = (group1_mean - group2_mean) / (group1_mean + group2_mean)

        return pd.DataFrame({
            f'{unique_groups[0]} Mean': [group1_mean],
            f'{unique_groups[1]} Mean': [group2_mean],
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
