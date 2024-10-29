from facesBenchmarkUtils.baseTask import *
from typing import Any, Callable

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
        true_label: str
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
        """
        super().__init__(
            name=name,
            pairs_file_path=pairs_file_path,
            images_path=images_path,
            distance_metric=distance_metric
        )
        if true_label not in self.pairs_df.columns:
            raise(Exception(f'{true_label} column does not exist in the input pairs file!'))
        else:
            self.true_label = true_label

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
        pairs_distances_df['similarity'] = 1 - pairs_distances_df['model_computed_distance']
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