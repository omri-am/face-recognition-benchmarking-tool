from .baseTask import *
import matplotlib.pyplot as plt
import seaborn as sns

class AccuracyTask(BaseTask):
    """
    A task that evaluates the accuracy of the model.
    Attributes:
        true_label: Column name in the pairs file loaded, indicating whether the answer is correct.
    """
    def __init__(self, name: str, pairs_file_path: str, images_path: str, distance_metric, true_label: str = 'truth'):
        super().__init__(name=name, pairs_file_path=pairs_file_path, images_path=images_path, distance_metric=distance_metric)
        self.true_label = true_label

    def compute_task_performance(self, pairs_distances_df):
        pairs_distances_df['similarity'] = 1 - pairs_distances_df['nn_computed_distance']
        y_true = self.pairs_df[self.true_label]

        auc = roc_auc_score(y_true, pairs_distances_df['similarity'])

        fpr, tpr, thresholds = roc_curve(y_true, pairs_distances_df['similarity'])
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]

        y_pred = (pairs_distances_df['similarity'] > optimal_threshold).astype(int)
        accuracy = accuracy_score(y_true, y_pred)

        return pd.DataFrame(
            {'Accuracy': [round(accuracy, 5)],
            'Optimal Threshold': [round(optimal_threshold,5)],
            'AUC': [round(auc, 5)],
            'Distance Metric': [self.distance_metric.__name__]
            })
    
    def plot(output_dir, performances, *optional):
        for task_name, task_df in performances.groupby('Task Name'):
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Model Name', y='Accuracy', hue='Model Name', data=task_df)
            plt.title(f'Accuracy Comparison for Task: {task_name}')
            plt.ylabel('Accuracy')
            plt.xlabel('Model')
            plt.ylim(0, 1)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{task_name}_accuracy comparison.png'))
            plt.close()

class CorrelationTask(BaseTask):
    """
    A task that evaluates the correlation between the model's distance metric and the true labels.
    Attributes:
        correlation_metric: The correlation metric to be used for evaluating the task.
    """
    def __init__(self, name: str, pairs_file_path: str, images_path: str, distance_metric, correlation_metric = np.corrcoef):
        super().__init__(name=name, pairs_file_path=pairs_file_path, images_path=images_path, distance_metric=distance_metric)
        self.correlation_metric = correlation_metric

    def compute_task_performance(self, pairs_distances_df):
        correlation = self.correlation_metric(pairs_distances_df['nn_computed_distance'], self.pairs_df['distance'])[0, 1]
        return pd.DataFrame({'Correlation Score': [round(correlation, 5)],
                             'Distance Metric': [self.distance_metric.__name__],
                             'Correlation Metric':[self.correlation_metric.__name__]
                             })
    
    def plot(output_dir, performances, distances, *optional):
        for task_name, task_df in performances.groupby('Task Name'):
            for model_name, task_model_df in task_df.groupby('Model Name'):

                d_data = distances[model_name][task_name]

                correlation_score = task_model_df['Correlation Score'].values[0]
                plt.figure(figsize=(10, 8))
                
                sns.regplot(data=d_data, x='distance', y='nn_computed_distance', scatter=True, line_kws={'color': 'red'})                
                plt.title(f'Correlation Scatter Plot for Task: {task_name} ({model_name})\nCorrelation Score: {correlation_score:.2f}')
                plt.xlabel('Input File Distance')
                plt.ylabel('NN Computed Distance')
                plt.tight_layout()

                plt.savefig(os.path.join(output_dir, f'{task_name}-{model_name}_correlation scatter.png'))
                plt.close()
    
class RelativeDifferenceTask(BaseTask):
    """
    Attributes:
        group_column: Column name in the pairs filde loaded, distinguishing between the two groups in the file.
    """
    def __init__(self, name: str, pairs_file_path: str, images_path: str, distance_metric, group_column: str):
        super().__init__(name=name, pairs_file_path=pairs_file_path, images_path=images_path, distance_metric=distance_metric)
        self.group_column = group_column
    
    def compute_task_performance(self, pairs_distances_df):
        # Split DataFrame into two groups based on the column values (assuming binary or categorical values)
        group1 = pairs_distances_df[pairs_distances_df[self.group_column] == pairs_distances_df[self.group_column].unique()[0]]
        group2 = pairs_distances_df[pairs_distances_df[self.group_column] == pairs_distances_df[self.group_column].unique()[1]]

        # Compute the mean of each group (ignoring non-numeric columns)
        group1_mean = group1['nn_computed_distance'].mean()
        group2_mean = group2['nn_computed_distance'].mean()
        
        relative_difference = (group1_mean - group2_mean) / (group1_mean + group2_mean)
        return pd.DataFrame({
            'Group 1 Mean': [group1_mean],
            'Group 2 Mean': [group2_mean],
            'Relative Difference': [round(relative_difference, 5)],
            'Distance Metric': [self.distance_metric.__name__]
            })
    
    def plot(output_dir, performances, *optional):
        for task_name, task_df in performances.groupby('Task Name'):
            plt.figure(figsize=(12, 8))
            sns.barplot(x='Model Name', y='Relative Difference', hue='Model Name', data=task_df)
            plt.title(f'Relative Difference Comparison for Task: {task_name}')
            plt.ylabel('Relative Difference')
            plt.xlabel('Model')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{task_name}_relative-diff comparison.png'))
            plt.close()
