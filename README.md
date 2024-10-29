# Face Recognition Benchmarking Tool

Welcome to the **Face Recognition Benchmarking Tool**! This project is designed to help researchers easily work with PyTorch and run various experiments related to facial recognition. It serves as a comprehensive benchmarking tool for face-related tasks, allowing for easy integration of models and tasks, and facilitating the computation of distances between tensors based on input pairs.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Architecture](#architecture)
  - [BaseModel](#basemodel)
    - [Implemented Models](#implemented-models)
    - [Extending BaseModel](#extending-basemodel)
  - [BaseTask](#basetask)
    - [Implemented Tasks](#implemented-tasks)
    - [Extending BaseTask](#extending-basetask)
  - [MultiModelTaskManager](#multimodeltaskmanager)
  - [PlotHelper](#plothelper)
- [Getting Started](#getting-started)
  - [Installation](#installation)
  - [Directory Structure](#directory-structure)
  - [Usage](#usage)
    - [Example: Running Experiments](#example-running-experiments)
    - [Example: Implementing Custom Models and Tasks](#example-implementing-custom-models-and-tasks)
    - [Example: Using LLM Propmpts to Implement Models and Tasks](#example-useful-llm-propts-for-implementation)
  - [Exported Results](#exported-results)
    - [CSV Files](#csv-files)
    - [Plots](#plots)
    - [Output Directory Structure](#output-directory-structure)
    - [Accessing and Interpreting Results](#accessing-and-interpreting-results)
    - [Customization](#customization)
- [Contact](#contact)

## Introduction

The Face Recognition Benchmarking Tool simplifies the process of experimenting with different facial recognition models and tasks. By providing abstract base classes and utilities, it enables researchers to:

- Easily integrate new models and tasks.
- Run multiple models across various tasks simultaneously.
- Compute distances and metrics efficiently.
- Benchmark and compare model performances on face-related tasks.
- Generate insightful plots for analysis.

## Features

- **Modular Architecture**: Abstract base classes for models and tasks facilitate easy extension.
- **Pre-implemented Models**: Includes implementations of popular models like VGG16, DINO, and CLIP.
- **Task Management**: Run multiple tasks such as accuracy computation, correlation analysis, and more.
- **Multi-Model Execution**: Execute multiple models across different tasks seamlessly.
- **Customizable**: Users can implement their own models and tasks by extending the base classes.
- **Plotting Utilities**: Built-in support for generating insightful plots for analysis.

## Architecture

The project is composed of four main components:

1. **BaseModel**: An abstract class that provides foundational functionalities for neural network models.
2. **BaseTask**: An abstract class representing tasks to be run in the benchmark.
3. **MultiModelTaskManager**: Manages the execution of multiple models across different tasks.
4. **PlotHelper**: A utility class for generating plots from task results.

### BaseModel

The `BaseModel` class serves as an abstraction for neural network models. It encapsulates common functionalities, such as model building, weight loading, preprocessing, and forward pass execution.

#### Key Features

- **Model Building**: Abstract method `_build_model` to define the architecture.
- **Weight Loading**: Handles loading pre-trained weights.
- **Layer Extraction**: Allows hooking into specific layers to extract intermediate outputs.
- **Preprocessing**: Provides default preprocessing or allows custom preprocessing functions.
- **Device Management**: Automatically utilizes GPU if available.

#### Implemented Models

The following models have been implemented by extending `BaseModel`:

1. **Vgg16Model**: Uses the VGG16 architecture.
2. **DinoModel**: Utilizes the DINO Transformer-based model.
3. **CLIPModel**: Incorporates the CLIP model for image embeddings.

#### Extending BaseModel

To implement a new model, you need to create a subclass of `BaseModel` and implement the following abstract methods:

1. `_build_model()`: Define the model architecture.
2. `_forward(input_tensor)`: Define the forward pass logic.

##### Example: Implementing a Custom Model

```python
class CustomModel(BaseModel):
    def __init__(self, name, weights_path=None, extract_layers=None, preprocess_function=None):
        super().__init__(name=name, weights_path=weights_path, extract_layers=extract_layers, preprocess_function=preprocess_function)

    def _build_model(self):
        # Define your custom model architecture
        self.model = YourCustomModel()

    def _forward(self, input_tensor):
        # Define the forward pass
        if input_tensor.ndim == 3:
            input_tensor = input_tensor.unsqueeze(0)
        return self.model(input_tensor)

    def preprocess_image(self, image_path):
        # Define custom preprocessing if needed
        image = Image.open(image_path).convert('RGB')
        return self.preprocess(image)
```

**Notes:**

- Ensure that your model inherits from `nn.Module`.
- If you need custom preprocessing, override `preprocess_image` or pass a `preprocess_function` during initialization.
- Use `extract_layers` if you want to hook into specific layers.

### BaseTask

The `BaseTask` class is an abstraction for tasks that can be run in the benchmarking process. Each task defines a specific computation or analysis to be performed on model outputs.

#### Key Features

- **Task Definition**: Abstract methods to define task-specific logic.
- **Distance Computation**: Handles computation of distances between embeddings.
- **Pair Input Handling**: Loads and processes input pairs for evaluation.
- **Validation**: Ensures the correctness of distance metrics and input paths.

#### BaseTask Class Structure

```python
class BaseTask(ABC):
    def __init__(
        self,
        name: str,
        pairs_file_path: str,
        images_path: str,
        distance_metric: Callable[[Any, Any], float] = pairwise.cosine_distances
    ) -> None:
        self.name = name
        self.pairs_file_path = pairs_file_path
        self.pairs_df = self.__load_file(pairs_file_path)
        self.images_path = self.__validate_path(images_path)
        self.distance_metric, self.distance_metric_name = self.__validate_and_set_distance_metric(distance_metric)

    @abstractmethod
    def compute_task_performance(self, pairs_distances_df: pd.DataFrame) -> pd.DataFrame:
        pass

    @abstractmethod
    def plot(self, output_dir: str, performances: pd.DataFrame, *optional: Any):
        pass
```

#### Implemented Tasks

The following tasks have been implemented by extending `BaseTask`:

1. **AccuracyTask**: Computes accuracy, AUC, and optimal thresholds.
2. **CorrelationTask**: Computes correlation between model outputs and given distances.
3. **ConditionedAverageDistances**: Computes average distances under certain conditions.
4. **RelativeDifferenceTask**: Calculates relative differences, useful for effects like the Thatcher effect.

##### AccuracyTask

Computes the accuracy of the model by comparing the predicted similarities against true labels.

**Requirements:**

- **Pairs File Columns**:
  - `'img1'`, `'img2'`: Image filenames.
  - **Truth Label Column**: By default `'truth'`, but can be specified using the `true_label` parameter.
- **Parameters**:
  - `true_label`: The name of the column in the pairs file that contains the ground truth labels (e.g., `'same'`, `'match'`).

##### CorrelationTask

Computes the correlation between the model-computed distances and given distances.

**Requirements:**

- **Pairs File Columns**:
  - `'img1'`, `'img2'`: Image filenames.
  - `'distance'`: The ground truth distances between image pairs.
- **Parameters**:
  - `correlation_metric`: A callable to compute the correlation (e.g., `np.corrcoef`, `spearmanr`).

##### ConditionedAverageDistances

Computes the average distances for different conditions specified in the pairs file.

**Requirements:**

- **Pairs File Columns**:
  - `'img1'`, `'img2'`: Image filenames.
  - **Condition Column**: By default `'condition'`, but can be specified using the `condition_column` parameter.
- **Parameters**:
  - `condition_column`: The name of the column that specifies the condition for each pair.
  - `normalize`: Whether to normalize the distances (default is `True`).

##### RelativeDifferenceTask

Calculates the relative difference between two groups, useful for tasks like evaluating the Thatcher effect.

**Requirements:**

- **Pairs File Columns**:
  - `'img1'`, `'img2'`: Image filenames.
  - **Group Column**: Specified using the `group_column` parameter. This column must have exactly two unique values.
- **Parameters**:
  - `group_column`: The name of the column that specifies the group for each pair.

#### Extending BaseTask

To create a new task, you need to subclass `BaseTask` and implement the required abstract methods.
Altough this is an open-source project, it is highly recommended to use the already implemented tasks. You can always [contact](#contact) us for further requests and questions.

<details>
  <summary>##### Example: Implementing a Custom Task</summary>

  ```python
  class CustomTask(BaseTask):
      def __init__(
          self,
          name: str,
          pairs_file_path: str,
          images_path: str,
          distance_metric: Callable[[Any, Any], float],
          **kwargs
      ) -> None:
          super().__init__(
              name=name,
              pairs_file_path=pairs_file_path,
              images_path=images_path,
              distance_metric=distance_metric
          )
          # Initialize any additional attributes here

      def compute_task_performance(self, pairs_distances_df: pd.DataFrame) -> pd.DataFrame:
          # Implement the computation logic using pairs_distances_df
          # Return a DataFrame with performance metrics
          pass

      @staticmethod
      def plot(
          output_dir: str,
          performances: pd.DataFrame,
          *optional: Any
      ) -> None:
          # Implement plotting logic
          pass
  ```

  **Notes:**

  - The `compute_task_performance` method should contain the main logic of your task and return a DataFrame with the computed metrics.
  - The `plot` method should handle the visualization of results.
  - Ensure that your pairs file contains all the necessary columns required by your custom task.

</details>

### MultiModelTaskManager

The `MultiModelTaskManager` orchestrates the execution of multiple models across different tasks. It handles loading models, running tasks, computing distances, and managing results.

#### Key Features

- **Model Management**: Loads and manages multiple models.
- **Task Execution**: Runs all models on all implemented tasks.
- **Distance Computation**: Computes distances between tensors as per the input pairs.
- **Result Aggregation**: Collects and organizes performance metrics and distances.
- **Export Utilities**: Exports computed metrics and unified summaries to CSV files.
- **Parallel Processing**: Utilizes DataLoader and batching for efficient computation.

#### Key Methods

- `add_tasks`: Adds tasks to the manager.
- `add_models`: Adds models to the manager.
- `run_task`: Runs a specific task with a specific model.
- `run_all_tasks_with_model`: Runs all tasks with a specific model.
- `run_all_tasks_all_models`: Runs all tasks with all models.
- `export_computed_metrics`: Exports computed distances and metrics to CSV files.
- `export_model_results_by_task`: Exports results and plots for each task.
- `export_unified_summary`: Exports a unified summary of all results.

### PlotHelper

The `PlotHelper` class provides static methods for generating plots from the task results. It supports bar plots and scatter plots, facilitating visual analysis.

#### Key Features

- **Bar Plots**: Compare performance metrics across models and layers.
- **Scatter Plots**: Visualize correlations between computed distances and true distances.
- **Customization**: Adjust plot sizes, labels, titles, and more.

#### PlotHelper Methods

- `bar_plot`: Generates a bar plot for performance metrics.
- `scatter_plot`: Generates scatter plots for correlation analyses.

## Getting Started

### Installation

To get started with the Face Recognition Benchmarking Tool, follow these steps:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/your-username/face-recognition-benchmarking-tool.git
   cd face-recognition-benchmarking-tool
   ```

2. **Install Dependencies**

   Install the required Python packages using `pip` or your desired package installer:

   ```bash
   pip install torch torchvision pandas numpy matplotlib scikit-learn Pillow tqdm transformers
   ```

   Make sure you have Python 3.7 or higher installed.

3. **Organize the Directory Structure**

   The project is organized into several directories, each serving a specific purpose. Upon cloning/downloading the repository, the below would be the structure of the project on your machine. This is the reccommended structure so you could easily follow these instructions. 

   ```plaintext
   face-recognition-benchmarking-tool/
   ├── models/
   │   ├── __init__.py
   │   ├── vgg16Model.py
   │   ├── dinoModel.py
   │   └── clipModel.py
   ├── tasks/
   │   ├── __init__.py
   │   ├── accuracyTask.py
   │   ├── correlationTask.py
   │   ├── conditionedAverageDistances.py
   │   └── relativeDifferenceTask.py
   ├── facesBenchmarkUtils/
   │   ├── __init__.py
   │   ├── baseModel.py
   │   ├── baseTask.py
   │   └── multiModelTaskManager.py
   ├── benchmark_runner.py
   ├── README.md
   └── requirements.txt
   ```

   **Notes:**

   - The `__init__.py` files allow for easier imports within the package.
   - The `benchmark_runner.py` script is where you can define and run your experiments.

4. **Set Up Your Data**

   - Place your datasets inside the project's directory.
   
   Example with the LFW dataset:
   ```plaintext
   face-recognition-benchmarking-tool/
   ├── models/
   ├── tasks/
   ├── facesBenchmarkUtils/
   ├── tests_datasets/
   │   ├── LFW
   │   │   ├── lfw_test_pairs_only_img_names.txt
   │   │   ├── lfw-align-128
   │   │   │   ├── Abel_Pacheco
   │   │   │   │   ├── Abel_Pacheco_0001.jpg
   │   │   │   │   ├── Abel_Pacheco_0002.jpg
   │   │   │   │   ├── ...
   │   │   │   ├── Edward_Kennedy
   │   │   │   │   ├── Edward_Kennedy_0001.jpg
   │   │   │   │   ├── ...
   │   │   │   ├── Mathilda_Karel_Spak
   │   │   │   │   ├── Mathilda_Karel_Spak_0001.jpg
   │   │   │   │   ├── ...
   │   │   │   ├── ...
   ```
   - Ensure that the pairs files contain the proper images path.

   Example with the LFW dataset:
   ```plaintext
   img1,img2,same
   Abel_Pacheco/Abel_Pacheco_0001.jpg,Abel_Pacheco/Abel_Pacheco_0004.jpg,1
   Edward_Kennedy/Edward_Kennedy_0001.jpg,Mathilda_Karel_Spak/Mathilda_Karel_Spak_0001.jpg,0
   ...
   ```

### Directory Structure

Here's a brief overview of the main directories:

- **models/**: Contains model implementations extending `BaseModel`.
  - `vgg16Model.py`
  - `dinoModel.py`
  - `clipModel.py`
- **tasks/**: Contains task implementations extending `BaseTask`.
  - `accuracyTask.py`
  - `correlationTask.py`
  - `conditionedAverageDistances.py`
  - `relativeDifferenceTask.py`
- **facesBenchmarkUtils/**: Contains utility classes and the `MultiModelTaskManager`.
  - `baseModel.py`
  - `baseTask.py`
  - `multiModelTaskManager.py`
- **tests_datasets/**: Contains datasets for testing and experiments, organized by task.
- **benchmark_runner.py**: Script to define and run your experiments.

### Usage

#### Example: Running Experiments

Below is an example of how to set up models and tasks, integrate them with the `MultiModelTaskManager`, and run experiments.

```python
def main():
    # Import necessary modules
    from models.vgg16Model import Vgg16Model
    from models.clipModel import CLIPModel
    from models.dinoModel import DinoModel
    from tasks.accuracyTask import AccuracyTask
    from tasks.correlationTask import CorrelationTask
    from facesBenchmarkUtils.multiModelTaskManager import MultiModelTaskManager
    import os

    # Initialize models
    vgg16_trained = Vgg16Model(
        name='VGG16-trained',
        weights_path='/path/to/trained_weights.pth',
        extract_layers=['avgpool', 'classifier.5']
    )
    vgg16_untrained = Vgg16Model(
        name='VGG16-untrained',
        extract_layers=['avgpool', 'classifier.5']
    )
    clip_model = CLIPModel(name='CLIP')
    dino_model = DinoModel(name='DINO', version='facebook/dinov2-base')

    # Initialize tasks
    lfw_pairs = '/path/to/lfw_pairs.txt'
    lfw_images = '/path/to/lfw_images'

    lfw_accuracy_task = AccuracyTask(
        name='LFW Accuracy',
        pairs_file_path=lfw_pairs,
        images_path=lfw_images,
        true_label='same',
        distance_metric=batch_cosine_distance
    )

    # Additional tasks can be initialized similarly

    # Initialize the MultiModelTaskManager
    manager = MultiModelTaskManager(
        models=[vgg16_trained, vgg16_untrained, clip_model, dino_model],
        tasks=[lfw_accuracy_task],
        batch_size=32
    )

    # Run all tasks with all models
    export_path = 'path/to/export/results/
    output_dir = os.path.join(os.getcwd(), export_path)
    manager.run_all_tasks_all_models(export_path=output_dir, print_log=True)

    # Access results
    performances = manager.tasks_performance_dfs
    distances = manager.model_task_distances_dfs

    # Export unified summary
    manager.export_unified_summary(export_path=output_dir)

if __name__ == '__main__':
    main()
```

**Notes:**

- Replace `/path/to/...` with actual paths to your datasets, images, and model weights.
- Ensure that the pairs files and images directories exist and are correctly formatted.
- The `batch_cosine_distance` function should be defined or imported as per your implementation.

#### Example: Implementing Custom Models and Tasks

You can extend the functionality by creating custom models and tasks.

##### Implementing a Custom Model

```python
class MyCustomModel(BaseModel):
    def _build_model(self):
        # Define custom architecture
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 222 * 222, 128)
        )

    def _forward(self, input_tensor):
        return self.model(input_tensor)

    def preprocess_image(self, image_path):
        image = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])
        return transform(image)
```

##### Implementing a Custom Task

```python
class MyCustomTask(BaseTask):
    def compute_task_performance(self, pairs_distances_df: pd.DataFrame) -> pd.DataFrame:
        # Custom computation logic
        metric_value = pairs_distances_df['model_computed_distance'].mean()
        return pd.DataFrame({'Custom Metric': [metric_value]})

    @staticmethod
    def plot(output_dir: str, performances: pd.DataFrame, *optional: Any) -> None:
        # Custom plotting logic
        PlotHelper.bar_plot(
            performances=performances,
            y='Custom Metric',
            ylabel='Custom Metric',
            title_prefix='Custom Task Results',
            output_dir=output_dir,
            file_name='custom_task_results'
        )
```

##### Running with Custom Models and Tasks

```python
def main():
    # Initialize custom model and task
    custom_model = MyCustomModel(name='CustomModel')
    custom_task = MyCustomTask(
        name='CustomTask',
        pairs_file_path='/path/to/pairs.csv',
        images_path='/path/to/images',
        distance_metric=your_custom_distance_function
    )

    # Initialize the manager
    manager = MultiModelTaskManager(
        models=[custom_model],
        tasks=[custom_task],
        batch_size=16
    )

    # Run the custom task with the custom model
    manager.run_all_tasks_all_models(export_path='/path/to/output', print_log=True)

if __name__ == '__main__':
    main()
```

**Notes:**

- Ensure that any custom distance functions or metrics are correctly implemented and validated.
- Adjust batch sizes and other parameters as needed for your computational resources.

#### Example: Useful LLM Propts For Implementation

### Running the MultiModelTaskManager on Given Tasks and Models

**Prompt:**

> I am working on a Python project called the Face Recognition Benchmarking Tool, which allows me to run multiple neural network (NN) models on different tasks easily. All the needed files are in the same folder as my python file. I need assistance in creating a Python script (run_benchmark.py) that:
>
> Imports all necessary classes and modules.
> Initializes the models (Vgg16Model, DinoModel, CLIPModel) with appropriate parameters, including loading weights if provided.
> Initializes the AccuracyTask (and any other tasks) with the provided image paths and distance metrics.
> Creates an instance of MultiModelTaskManager, adding all models and tasks to it.
> Runs all tasks on all models using the manager.
> Exports the computed metrics to specified paths.
> File Paths:

> Image path: 'path/to/img/folder'
> Pairs file path: 'path/to/pair_file.csv'
> Weights path: 'path/to/weights_file.pth'
> Classes with Docstrings:
<details>
  <summary>docstrings and class signature for the desired models, tasks and MultiModelTaskManager</summary>
  
  ```python
  class Vgg16Model(BaseModel):
      """
      A VGG16 model implementation for face recognition tasks.

      This class initializes a VGG16 model, optionally loading pre-trained weights.
      It allows extraction of specific layers and provides methods for preprocessing
      images and forwarding inputs through the model.

      Attributes
      ----------
      name : str
          The name of the model.
      weights_path : str or None
          Path to the model's weights file (.pth extention). If None, default pre-trained weights are used.
      extract_layers : str or list of str
          Layer(s) from which to extract outputs.
      preprocess_function : callable or None
          Function to preprocess input images.
      num_identities : int or None
          Number of identities (classes) in the model, set if weights are loaded.
      model : torch.nn.Module
          The VGG16 neural network model.
      device : torch.device
          The device (CPU or GPU) on which the model is placed.
      hook_outputs : dict
          Dictionary to store outputs from hooked layers.
      """

      def __init__(
          self,
          name: str,
          weights_path: Optional[str] = None,
          extract_layers: Optional[Union[str, List[str]]] = 'classifier.3',
          preprocess_function: Optional[Callable[[Any], Any]] = None
      ):

  class DinoModel(BaseModel):
      """
      A DINO model implementation for face recognition tasks.

      This class initializes a DINO model using the specified version, handles image
      preprocessing, and provides methods for forwarding inputs through the model.

      Attributes
      ----------
      name : str
          The name of the model.
      version : str
          The version identifier for the DINO model.
      model : torch.nn.Module
          The DINO neural network model.
      processor : transformers.AutoImageProcessor
          The image processor for preparing inputs.
      device : torch.device
          The device (CPU or GPU) on which the model is placed.
      hook_outputs : dict
          Dictionary to store outputs from hooked layers.
      """

      def __init__(
          self, 
          name: str, 
          version: str = 'facebook/dinov2-base'
      ):

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

  class MultiModelTaskManager:
      """
      Manages multiple models and tasks, facilitating the computation of task performance across models.

      Attributes
      ----------
      tasks : Dict[str, BaseTask]
          Dictionary of task instances, keyed by task name.
      models : Dict[str, BaseModel]
          Dictionary of model instances, keyed by model name.
      model_task_distances_dfs : Dict[str, Dict[str, pd.DataFrame]]
          Nested dictionary storing computed distances for each model and task.
      tasks_performance_dfs : Dict[str, pd.DataFrame]
          Dictionary storing performance DataFrames for each task.
      batch_size : int
          Batch size for data loading.

      Methods
      -------
      add_tasks(tasks: Union[BaseTask, List[BaseTask]]) -> None
          Adds tasks to the manager.
      add_models(models: Union[BaseModel, List[BaseModel]]) -> None
          Adds models to the manager.
      export_computed_metrics(export_path: str) -> None
          Exports computed metrics for each model and task to CSV files.
      export_model_results_by_task(export_path: str) -> None
          Exports the model results grouped by task type and generates plots.
      export_unified_summary(export_path: str = os.getcwd()) -> None
          Exports a unified summary CSV file with performance metrics.
      run_task(model_name: str, task_name: str, export_path: str = os.getcwd(), print_log: bool = False) -> None
          Runs a specific task on a specific model.
      run_task_with_all_models(task_name: str, export_path: str = os.getcwd(), print_log: bool = False) -> None
          Runs a specific task on all models and exports the results.
      run_all_tasks_with_model(model_name: str, export_path: str = os.getcwd(), print_log: bool = False) -> None
          Runs all tasks on a specific model and exports the results.
      run_all_tasks_all_models(export_path: str = os.getcwd(), print_log: bool = False) -> None
          Runs all tasks on all models and exports the results.
      """

      def __init__(
          self,
          models: Union[BaseModel, List[BaseModel]],
          tasks: Union[BaseTask, List[BaseTask]],
          batch_size: int = 32
      ) -> None:
  ```
</details>

> Requirements:
>
> Implement run_benchmark.py following the structure outlined above.
> Ensure proper error handling and logging.
> Optionally, handle configurations via command-line arguments.
> Make sure all dependencies are correctly imported and utilized.
> Please generate the run_benchmark.py script based on the provided information.

<details>
  <summary>ChatGPT Example Result</summary>

  ```python
  import pandas as pd
  from your_module import Vgg16Model, DinoModel, AccuracyTask, MultiModelTaskManager

  # Define paths to the required files
  img_path = 'path/to/img/folder'
  pairs_file_path = 'path/to/pair_file.csv'
  weights_path = 'path/to/weights_file.pth'

  # Create model instances
  vgg_model = Vgg16Model(
      name='VGG16_Face_Recognition',
      weights_path=weights_path,
      extract_layers='classifier.3'  # Specify which layer's output to extract
  )

  dino_model = DinoModel(
      name='DINO_Face_Recognition',
      version='facebook/dinov2-base'  # Specify the version of DINO
  )

  # Create a distance metric function (you can replace this with the actual implementation)
  def example_distance_metric(a, b):
      # Replace with actual distance metric calculation
      return ((a - b) ** 2).sum() ** 0.5

  # Create task instance
  accuracy_task = AccuracyTask(
      name='Accuracy_Evaluation',
      pairs_file_path=pairs_file_path,
      images_path=img_path,
      distance_metric=example_distance_metric,
      true_label='true_label_column'  # Replace with the actual true label column name
  )

  # Initialize the MultiModelTaskManager
  task_manager = MultiModelTaskManager(
      models=[vgg_model, dino_model],
      tasks=[accuracy_task],
      batch_size=32
  )

  # Run all tasks on all models and export results
  task_manager.run_all_tasks_all_models(export_path='path/to/export_directory', print_log=True)

  # Optional: If you want to run a specific task with a specific model, uncomment below
  # task_manager.run_task(model_name='VGG16_Face_Recognition', task_name='Accuracy_Evaluation', export_path='path/to/export_directory', print_log=True)
  ```
</details>

#### Generating a New Model

**Prompt:** 

> I am working with a Python project called the Face Recognition Benchmarking Tool, which uses a BaseModel abstract class to define models. Here's the structure of BaseModel:
> ```python
> baseModel python code...
> ```
> I want to implement a new model called ResNetModel that extends BaseModel and uses a pre-trained ResNet50 architecture from torchvision.models. The model should:
> 
> Use the pre-trained weights.
> Implement the class' abstract methods.
> Implement custom preprocessing if necessary.
> Can you provide the Python code for the ResNetModel class, following the structure of BaseModel and ensuring it integrates smoothly with the existing code?

<details>
  <summary>ChatGPT Example Result</summary>

  ```python
  import torch
  import torch.nn as nn
  from torchvision import models, transforms
  from typing import Optional, List, Union, Callable, Any, Dict
  from PIL import Image

  class ResNetModel(BaseModel):
      def __init__(self, name: str, weights_path: Optional[str] = None,
                  extract_layers: Union[str, List[str]] = 'layer4',
                  preprocess_function: Optional[Callable[[Any], Any]] = None) -> None:
          super().__init__(name=name, weights_path=weights_path, extract_layers=extract_layers, preprocess_function=preprocess_function)

      def _build_model(self) -> None:
          model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if self.weights_path is None else None)

          # If the number of identities is set, modify the last layer
          if self.num_identities is not None:
              num_features = model.fc.in_features
              model.fc = nn.Linear(num_features, self.num_identities)
          else:
              self.num_identities = model.fc.out_features
          
          self.model = model

      def _forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
          if input_tensor.ndim == 3:
              input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension if necessary
          return self.model(input_tensor)

      def preprocess_image(self, image_path: str) -> torch.Tensor:
          """
          Preprocess an image given its file path.
          
          Parameters
          ----------
          image_path : str
              Path to the input image.
          
          Returns
          -------
          torch.Tensor
              Preprocessed image tensor.
          """
          image = Image.open(image_path).convert('RGB')
          return self.preprocess(image)
  ```
</details>

#### Generating a New Task
**Prompt:** 

> I am using the Face Recognition Benchmarking Tool, which has a BaseTask abstract class for defining tasks. Here is its structure:
> ```python
> baseTask python code...
> ```
> I want to implement a new task called MeanSquaredErrorTask that calculates the mean squared error (MSE) between the model-computed distances and true distances provided in the pairs file. The pairs file contains columns 'img1', 'img2', and 'true_distance'.  The task should: Compute the MSE between the model distances and 'true_distance'. Return a DataFrame with the MSE and model information. Provide a plotting function to visualize the error distribution. 
> 
> Can you provide the Python code for the MeanSquaredErrorTask class, ensuring it properly extends BaseTask and integrates with the existing code?

<details>
  <summary>ChatGPT Example Result</summary>

  ```python
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  import seaborn as sns
  from sklearn.metrics import mean_squared_error

  class MeanSquaredErrorTask(BaseTask):
      def __init__(self, name: str, pairs_file_path: str, images_path: str) -> None:
          super().__init__(name=name, pairs_file_path=pairs_file_path, images_path=images_path)

      def compute_task_performance(self, pairs_distances_df: pd.DataFrame) -> pd.DataFrame:
          """
          Computes the Mean Squared Error (MSE) between the model-computed distances and true distances.

          Parameters
          ----------
          pairs_distances_df : pandas.DataFrame
              DataFrame containing the computed distances for image pairs.

          Returns
          -------
          pd.DataFrame
              The performance metrics DataFrame with MSE and model information.
          """
          # Merge computed distances with the true distances from pairs_df
          merged_df = self.pairs_df.merge(pairs_distances_df, on=['img1', 'img2'], how='left')

          # Calculate MSE
          mse = mean_squared_error(merged_df['true_distance'], merged_df['model_computed_distance'], squared=True)
          
          # Create a DataFrame to return
          performance_df = pd.DataFrame({
              'Model Name': [self.name],
              'Mean Squared Error': [mse]
          })

          return performance_df

      def plot(self, output_dir: str, performances: pd.DataFrame, *optional: Any) -> None:
          """
          Visualizes the distribution of errors using a histogram.

          Parameters
          ----------
          output_dir : str
              Directory to save the plots.
          performances : pd.DataFrame
              DataFrame containing performance metrics.
          optional : Any
              Additional parameters for customization if needed.
          """
          # Extracting model errors for plotting
          errors = performances['Mean Squared Error'].values
          
          plt.figure(figsize=(10, 6))
          sns.histplot(errors, bins=20, kde=True)
          plt.title(f'Error Distribution for {self.name}', fontsize=SUPTITLE_SIZE)
          plt.xlabel('Mean Squared Error', fontsize=SUBTITLE_SIZE)
          plt.ylabel('Frequency', fontsize=SUBTITLE_SIZE)
          
          # Save the plot
          plt.tight_layout()
          plt.savefig(os.path.join(output_dir, f'{self.name}_error_distribution.png'))
          plt.close()
  ```
</details>

#### Generating a Custom Plot Function
**Prompt:**
> I am creating a subclass of a class that has an abstract static plot method with the following signature: 
> ```python
> @staticmethod
> def plot(output_dir, performances, *optional)
> ```
> I want to create a plotting function that generates a histogram of computed distances. The dataframe that are available to me are performances, which contains a final computed metric, and distances, which contains for each pair of images their computed distance. Can you help me modify the plot method in CustomTask to include this functionality? 

<details>
  <summary>ChatGPT Example Result</summary>

  ```python
  import matplotlib.pyplot as plt
  import pandas as pd

  class CustomTask(YourBaseClass):  # Replace YourBaseClass with the actual base class name
      @staticmethod
      def plot(output_dir, performances, distances, *optional):
          """
          Generate a histogram of computed distances.

          Args:
              output_dir (str): Directory to save the plot.
              performances (pd.DataFrame): Summary data.
              distances (pd.DataFrame): DataFrame containing computed distances for each pair of images.
              *optional: Additional optional parameters.
          """
          # Ensure distances DataFrame has the expected structure
          if distances.empty:
              print("No distance data available to plot.")
              return

          # Extract distance values
          distance_values = distances.values.flatten()  # Flatten if it's a 2D array

          # Create the histogram
          plt.figure(figsize=(10, 6))
          plt.hist(distance_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
          plt.title('Histogram of Computed Distances')
          plt.xlabel('Distance')
          plt.ylabel('Frequency')
          
          # Save the plot
          plt.savefig(f"{output_dir}/histogram_of_distances.png")
          plt.close()

          print(f"Histogram saved to {output_dir}/histogram_of_distances.png")

  # Example usage:
  # CustomTask.plot('output_directory', performances_dataframe, distances_dataframe)
  ```
</details>

## Exported Results

The Face Recognition Benchmarking Tool generates various CSV files and plots as part of the evaluation process. These outputs provide detailed insights into model performances across different tasks and facilitate comprehensive analysis.

### CSV Files

After running experiments, the tool exports several CSV files containing the computed metrics and distances. These files are saved in the specified output directory and are organized per model and task.

1. **Pairwise Distances CSVs**:
   - **Filename Format**: `model_name_task_name_pairs_distances.csv`
   - **Description**: Contains the computed distances between image pairs for each model and task combination. The CSV includes the following columns:
     - `img1`: Filename of the first image in the pair.
     - `img2`: Filename of the second image in the pair.
     - `model_computed_distance`: The distance between the embeddings of `img1` and `img2` as computed by the model.
     - Additional columns from the original pairs file (e.g., labels, conditions).

2. **Task Performance Metrics CSVs**:
   - **Filename Format**: `model_name_task_name_performance_metrics.csv`
   - **Description**: Contains the performance metrics computed for each model and task. The metrics vary depending on the task but may include accuracy, AUC, correlation coefficients, mean distances, etc. The CSV includes columns such as:
     - `Task Name`: Name of the task.
     - `Model Name`: Name of the model.
     - `Layer Name`: Name of the layer (if layer-wise analysis is performed).
     - Metric columns specific to the task (e.g., `Accuracy`, `AUC`, `Correlation Score`).

3. **Unified Summary CSV**:
   - **Filename**: `models_unified_results.csv`
   - **Description**: Provides a consolidated view of all models and tasks in a single CSV file. Each row corresponds to a model and layer, and columns represent the performance metrics from different tasks. This summary facilitates easy comparison across models and tasks.
   - **Columns**:
     - `Model Name`
     - `Layer Name`
     - Task-specific metric columns (e.g., `AccuracyTask: Accuracy`, `CorrelationTask: Correlation Score`).

### Plots

The tool generates various plots to visualize the performance of models across tasks. These plots are saved in the output directory, typically in PNG format.

1. **Bar Plots**:
   - **Purpose**: Compare performance metrics across models and layers.
   - **Generated For**:
     - Accuracy scores.
     - Correlation coefficients.
     - Mean values under different conditions.
     - Relative differences between groups.
   - **Filename Format**: `task_name_metric_name_comparison.png`
   - **Description**: Displays a bar chart where the x-axis represents the models (and layers, if applicable), and the y-axis represents the metric value. Useful for quickly assessing which models perform better on a specific task.

2. **Scatter Plots**:
   - **Purpose**: Visualize the relationship between model-computed distances and true distances or human-annotated scores.
   - **Generated For**:
     - Correlation tasks where the relationship between computed distances and true distances is analyzed.
   - **Filename Format**: `task_name_scatters_comparison.png`
   - **Description**: Plots each pair's computed distance against the true distance, allowing visualization of the correlation between them.

3. **Conditioned Plots**:
   - **Purpose**: Analyze model performance under different conditions or groups.
   - **Generated For**:
     - Tasks that involve conditions or group comparisons (e.g., `ConditionedAverageDistances`, `RelativeDifferenceTask`).
   - **Filename Format**: `task_name_condition_name_comparison.png`
   - **Description**: Bar plots that show performance metrics separately for each condition or group, facilitating the analysis of how models behave under different scenarios.

### Output Directory Structure

The results are organized in a structured manner within the specified output directory:

- **Root Output Directory**:
  - Contains the unified summary CSV (`models_unified_results.csv`).
  - Contains plots that compare models across tasks.

- **Per-Task Subdirectories**:
  - Named after each task (e.g., `AccuracyTask`, `CorrelationTask`).
  - Contain CSVs and plots specific to that task.
  - **Files Included**:
    - Pairwise distances CSVs.
    - Task performance metrics CSVs.
    - Task-specific plots.

### Accessing and Interpreting Results

- **CSV Files**:
  - Can be opened using spreadsheet software or analyzed programmatically using tools like pandas.
  - Enable detailed examination of individual model performances and pairwise comparisons.

- **Plots**:
  - Provide visual summaries that can be included in reports or presentations.
  - Help in quickly identifying trends, outliers, and performance differences.

### Customization

- The filenames and output paths can be customized by modifying the parameters in the `MultiModelTaskManager` and task classes.
- Additional plots or CSV exports can be implemented by extending the `plot` methods in the task classes or by processing the exported CSVs.

# Contact

For questions or support, please open an issue or contact:

Omri Amit, [omriamit@mail.tau.ac.il](mailto:omriamit@mail.tau.ac.il)

Mia Shlein, [miachaias@mail.tau.ac.il](mailto:miachaias@mail.tau.ac.il)