import pandas as pd
import matplotlib.pyplot as plt
import os
import math
import numpy as np

def plot_model_layer_superplots(df, model_name, layer_name, output_file_path):
    """
    Plots a superplot for a given model-layer combination, including all prefix-suffix combinations with data available.
    
    Args:
        df (pd.DataFrame): The dataframe containing the data.
        model_name (str): The name of the model.
        layer_name (str): The name of the layer.
        output_file_path (str): The directory where the plot should be saved.
    """
    # Extract all prefixes and columns
    all_prefixes = set([col.split(' - ')[0] for col in df.columns if ' - ' in col])
    all_columns = [col for col in df.columns if any(col.startswith(prefix + ' - ') for prefix in all_prefixes)]
    suffixes = set([col.split(': ')[1] for col in all_columns if ': ' in col])
    suffixes.remove('AUC')
    suffixes.remove('Optimal Threshold')

    # Filter dataframe for the given model and layer
    group_df = df[(df['Model Name'] == model_name) & (df['Layer Name'] == layer_name)]
    filtered_columns = [col for col in all_columns if col in group_df.columns]

    if filtered_columns:
        # Determine the actual number of prefix-suffix combinations with data
        prefix_suffix_combinations = []
        for prefix in all_prefixes:
            for suffix in suffixes:
                prefix_suffix_columns = [col for col in filtered_columns if col.startswith(prefix + ' - ') and col.endswith(': ' + suffix)]
                if prefix_suffix_columns:
                    prefix_suffix_combinations.append((prefix, suffix))

        # Create a superplot with subplots for each prefix-suffix combination that has data
        num_combinations = len(prefix_suffix_combinations)
        ncols = math.ceil(math.sqrt(num_combinations))
        nrows = math.ceil(num_combinations / ncols)
        
        # Set dynamic figsize and font sizes
        fig_width = min(8 * ncols, 30)  # Cap max width to avoid excessive size
        fig_height = min(6 * nrows, 20)  # Cap max height
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), constrained_layout=True)
        
        fig.suptitle(f'\n{model_name} - {layer_name}\n', fontsize=32)

        # Flatten axes if necessary
        axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

        # Plot each prefix-suffix combination in a separate subplot
        for plot_idx, (prefix, suffix) in enumerate(prefix_suffix_combinations):
            prefix_suffix_columns = [col for col in filtered_columns if col.startswith(prefix + ' - ') and col.endswith(': ' + suffix)]
            if prefix_suffix_columns:
                ax = axes[plot_idx]
                # Remove prefix and suffix from labels
                labels = [col.replace(prefix + ' - ', '').replace(': ' + suffix, '') for col in prefix_suffix_columns]
                group_df[prefix_suffix_columns].mean().plot(kind='bar', ax=ax)
                ax.set_title(f'{prefix} - {suffix}', fontsize=20)
                ax.set_ylabel('Mean Value', fontsize=14)
                ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=14)

        # Hide any empty subplots
        for idx in range(len(prefix_suffix_combinations), len(axes)):
            fig.delaxes(axes[idx])

        # Save the superplot
        plt.savefig(os.path.join(output_file_path, f'{model_name}_{layer_name}_taz.png'))
        plt.close()
        print(f'{model_name} - {layer_name}')
