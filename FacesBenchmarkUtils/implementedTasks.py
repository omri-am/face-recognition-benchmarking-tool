from .baseTask import *

class AccuracyTask(BaseTask):
    """
    A task that evaluates the accuracy of the model.
    Attributes:
        true_label: Column name in the pairs file loaded, indicating whether the answer is correct.
    """
    def __init__(self, name: str, pairs_file_path: str, images_path: str, distance_metric, true_label: str = 'truth'):
        super().__init__(name=name, pairs_file_path=pairs_file_path, images_path=images_path, distance_metric=distance_metric)
        self.true_label = true_label

    def compute_task_performance(self, pairs_df_with_calc):
        similarity = pairs_df_with_calc['nn_computed_distance'].apply(lambda x: 1-float(x))

        y_true = pairs_df_with_calc[self.true_label]
        auc = roc_auc_score(y_true, similarity)

        fpr, tpr, thresholds = roc_curve(y_true, similarity)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        y_pred = (similarity > optimal_threshold).astype(int)
        accuracy = accuracy_score(y_true, y_pred)

        return pd.DataFrame({"Accuracy": accuracy,
                             "Optimal Threshold": optimal_threshold,
                             "AUC": auc})

class CorrelationTask(BaseTask):
    """
    A task that evaluates the correlation between the model's distance metric and the true labels.
    Attributes:
        correlation_metric: The correlation metric to be used for evaluating the task.
    """
    def __init__(self, name: str, pairs_file_path: str, images_path: str, distance_metric, correlation_metric = np.corrcoef):
        super().__init__(name=name, pairs_file_path=pairs_file_path, images_path=images_path, distance_metric=distance_metric)

    def compute_task_performance(self, distances):
        correlation = self.correlation_metric(distances, self.pairs_df['distance'])[0, 1]
        return pd.DataFrame({"Correlation Score": correlation})