import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
import os

SUPTITLE_SIZE = 18
SUBTITLE_SIZE = 12

create_model_layer_column = lambda row: row['Model Name'] if row['Layer Name'] == 'default' else row['Model Name'] + ': ' + row['Layer Name']

def get_function_name(func):
    return getattr(func, '__name__', str(func))

class PlotHelper:
    """
    A helper class for generating plots related to model performance on tasks.
    """

    @staticmethod
    def bar_plot(performances, y, ylabel, title_prefix, output_dir, file_name, x=None, xlabel=None, ylim=None):
        """
        Creates and saves bar plots for task performances.

        Parameters
        ----------
        performances : pandas.DataFrame
            DataFrame containing performance metrics for each model and task.
        x : str
            The column name in `performances` to be used as the x-axis value.
        xlabel : str
            Label for the x-axis.
        y : str
            The column name in `performances` to be used as the y-axis value.
        ylabel : str
            Label for the y-axis.
        title_prefix : str
            Prefix for the plot titles.
        output_dir : str
            Directory where the plot images will be saved.
        file_name : str
            Base file name for the saved plot images.
        ylim : tuple of float, optional
            Y-axis limits for the plots as (min, max).

        Returns
        -------
        None
        """
        for task_name, task_df in performances.groupby('Task Name'):
            task_data = task_df.copy()
            task_data['Model-Layer'] = task_data.apply(create_model_layer_column, axis=1)

            width = max(round(len(task_data['Model-Layer'].unique()) / 20 * 12), 12)
            plt.figure(figsize=(width, 8))

            x = x if x else 'Model-Layer'
            xlabel = xlabel if xlabel else 'Model-Layer'

            ax = sns.barplot(x=x, y=y, data=task_data, hue=x)

            locations = ax.get_xticks()
            labels = [item.get_text() for item in ax.get_xticklabels()]
            ax.set_xticks(locations)
            ax.set_xticklabels(labels, rotation=45, ha='right')

            for p in ax.patches:
                ax.annotate(f'{p.get_height():.2f}',
                            (p.get_x() + p.get_width() / 2., p.get_height()),
                            ha='center', va='center', fontsize=14, color='black',
                            xytext=(0, 5), textcoords='offset points')

            plt.title(f'{title_prefix}: {task_name}', fontsize=SUPTITLE_SIZE)
            plt.ylabel(ylabel)
            plt.xlabel(xlabel)
            if ylim:
                plt.ylim(ylim)

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{task_name}_{file_name}.png'), bbox_inches='tight')
            plt.close()

    @staticmethod
    def scatter_plot(performances, distances, output_dir, file_name):
        """
        Creates and saves scatter plots showing the correlation between input distances and model-computed distances.

        Parameters
        ----------
        performances : pandas.DataFrame
            DataFrame containing performance metrics for each model and task.
        distances : dict
            Nested dictionary containing distance DataFrames for each model and task.
        output_dir : str
            Directory where the plot images will be saved.
        file_name : str
            Base file name for the saved plot images.

        Returns
        -------
        None
        """
        for task_name, task_df in performances.groupby('Task Name'):
            task_data = task_df.copy()
            task_data['Model-Layer'] = task_data.apply(create_model_layer_column, axis=1)

            num_plots = len(task_data)
            cols = math.ceil(math.sqrt(num_plots))
            rows = math.ceil(num_plots / cols)

            min_y = min(
                distances[model_name][task_name]['model_computed_distance'].min()
                for model_name in task_df['Model Name'].unique()
            )
            max_y = max(
                distances[model_name][task_name]['model_computed_distance'].max()
                for model_name in task_df['Model Name'].unique()
            )

            fig, axs = plt.subplots(rows, cols, figsize=(6 * cols, 5 * rows))
            axs = axs.flatten() if isinstance(axs, np.ndarray) else np.array([axs])

            idx = 0
            for _, row in task_data.iterrows():
                model_name = row['Model Name']
                layer_name = row['Layer Name']
                correlation_score = row['Correlation Score']

                d_data = distances[model_name][task_name]
                d_data_layer = d_data[d_data['layer_name'] == layer_name]

                ax = axs[idx]
                sns.regplot(
                    data=d_data_layer,
                    x='distance',
                    y='model_computed_distance',
                    scatter=True,
                    line_kws={'color': 'red'},
                    ax=ax
                )
                ax.set_title(f'{model_name} - {layer_name}\nCorrelation: {correlation_score:.2f}',
                             fontsize=SUBTITLE_SIZE)
                ax.set_xlabel('Input File Distance')
                ax.set_ylabel('NN Computed Distance')
                ax.set_ylim(max(min_y - 0.05, -1), min(max_y + 0.05, 1))
                idx += 1

            for i in range(idx, len(axs)):
                fig.delaxes(axs[i])

            plt.suptitle(f'Correlation Scatter Plots: {task_name}', fontsize=SUPTITLE_SIZE)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(os.path.join(output_dir, f'{task_name}_{file_name}.png'))
            plt.close()