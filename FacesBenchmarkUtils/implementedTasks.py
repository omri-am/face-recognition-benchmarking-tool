from .baseTask import *
import matplotlib.pyplot as plt
import seaborn as sns
import math

SUPTITLE_SIZE = 18
SUBTITLE_SIZE = 12

class PlotHelper():
    def bar_plot(performances, y, ylabel, title_prefix, output_dir, file_name, ylim=None):
        for task_name, task_df in performances.groupby('Task Name'):
            task_data = task_df.copy()
            task_data['Model-Layer'] = task_data['Model Name'] + ': ' + task_data['Layer Name']
            plt.figure(figsize=(12, 8))
            ax = sns.barplot(x='Model-Layer', y=y, data=task_data)
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.2f}',
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='baseline',
                            fontsize=11, color='black', xytext=(0, 3),
                            textcoords='offset points')
            plt.title(f'{title_prefix}: {task_name}', fontsize=SUPTITLE_SIZE)
            plt.ylabel(ylabel)
            plt.xlabel('Model-Layer')
            plt.ylim(ylim if ylim else None) 
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{task_name}_{file_name}.png'))
            plt.close()

    def scatter_plot(performances, distances, output_dir, file_name):
        for task_name, task_df in performances.groupby('Task Name'):
            task_data = task_df.copy()
            task_data['Model-Layer'] = task_data['Model Name'] + ': ' + task_data['Layer Name']

            num_plots = len(task_data)
            cols = math.ceil(math.sqrt(num_plots))
            rows = math.ceil(num_plots / cols)

            fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
            axs = axs.flatten()

            idx = 0
            for _, row in task_data.iterrows():
                model_name = row['Model Name']
                layer_name = row['Layer Name']
                correlation_score = row['Correlation Score']
                
                d_data = distances[model_name][task_name]
                d_data_layer = d_data[d_data['Layer Name'] == layer_name]
                
                ax = axs[idx]
                sns.regplot(data=d_data_layer,
                            x='distance',
                            y='nn_computed_distance',
                            scatter=True,
                            line_kws={'color': 'red'},
                            ax=ax)
                ax.set_title(f'{model_name} - {layer_name}\nCorrelation: {correlation_score:.2f}',
                            fontsize=SUBTITLE_SIZE)
                ax.set_xlabel('Input File Distance')
                ax.set_ylabel('NN Computed Distance')
                idx += 1
            
            for i in range(idx, len(axs)):
                fig.delaxes(axs[i])
            
            plt.suptitle(f'Correlation Scatter Plots: {task_name}', fontsize=SUPTITLE_SIZE)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(output_dir, f'{task_name}_{file_name}.png'))
            plt.close()

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
        PlotHelper.bar_plot(
            performances=performances,
            y='Accuracy',
            ylabel='Accuracy',
            ylim=(0,1),
            title_prefix='Accuracy Score Comparison',
            output_dir=output_dir,
            file_name='accuracy comparison')

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
        PlotHelper.scatter_plot(
            performances=performances,
            distances=distances,
            output_dir=output_dir,
            file_name='Scatters Comparison'
        )
    
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
        PlotHelper.bar_plot(
            performances=performances,
            y='Relative Difference',
            ylabel='Relative Difference',
            title_prefix='Relative Difference Comparison',
            output_dir=output_dir,
            file_name='Relative-Diff Comparison')