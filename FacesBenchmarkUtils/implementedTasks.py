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

    def compute_task_performance(self, distances):
        similarity = [1-float(x) for x in distances]

        y_true = self.pairs_df[self.true_label]
        auc = roc_auc_score(y_true, similarity)

        fpr, tpr, thresholds = roc_curve(y_true, similarity)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        y_pred = (similarity > optimal_threshold).astype(int)
        accuracy = accuracy_score(y_true, y_pred)

        return pd.DataFrame({"Accuracy": [round(accuracy, 5)],
                             "Optimal Threshold": [round(optimal_threshold,5)],
                             "AUC": [round(auc, 5)]})

class CorrelationTask(BaseTask):
    """
    A task that evaluates the correlation between the model's distance metric and the true labels.
    Attributes:
        correlation_metric: The correlation metric to be used for evaluating the task.
    """
    def __init__(self, name: str, pairs_file_path: str, images_path: str, distance_metric, correlation_metric = np.corrcoef):
        super().__init__(name=name, pairs_file_path=pairs_file_path, images_path=images_path, distance_metric=distance_metric)
        self.correlation_metric = correlation_metric

    def compute_task_performance(self, distances):
        correlation = self.correlation_metric(distances, self.pairs_df['distance'])[0, 1]
        return pd.DataFrame({"Correlation Score": [round(correlation, 5)]})