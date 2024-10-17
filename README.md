# Benchmarking Framework for Neural Network Models

This repository provides a benchmarking framework for machine learning researchers to evaluate and compare the performance of neural network models on various tasks, specifically focusing on face recognition and related tasks. The framework allows you to:

- **Define custom models and tasks**
- **Compute performance metrics across multiple models and tasks**
- **Visualize results through plots**
- **Export computed metrics and summaries**

## Table of Contents

- [Introduction](#introduction)
- [Architecture](#architecture)
  - [BaseModel](#basemodel)
  - [BaseTask](#basetask)
  - [Derived Task Classes](#derived-task-classes)
    - [AccuracyTask](#accuracytask)
    - [CorrelationTask](#correlationtask)
    - [RelativeDifferenceTask](#relativedifferencetask)
  - [PlotHelper](#plothelper)
  - [MultiModelTaskManager](#multimodeltaskmanager)
  - [PairDataset](#pairdataset)
- [Usage](#usage)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Defining Models](#defining-models)
  - [Defining Tasks](#defining-tasks)
  - [Running Tasks](#running-tasks)
  - [Visualizing Results](#visualizing-results)
- [Examples](#examples)
  - [Example: Running an Accuracy Task](#example-running-an-accuracy-task)
  - [Example: Running All Tasks on All Models](#example-running-all-tasks-on-all-models)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The framework is designed to streamline the process of evaluating neural network models on various tasks. It provides an abstract base for models and tasks, allowing you to plug in your implementations and compare performance across different scenarios.

Key features include:

- **Modular Design**: Easily add new models and tasks by extending base classes.
- **Parallel Processing**: Run multiple tasks and models efficiently.
- **Extensive Documentation**: Classes and methods are well-documented for ease of use.
- **Visualization Tools**: Generate plots to visualize performance metrics.

## Architecture

The framework is built with several core components:

### BaseModel

An abstract base class representing a neural network model. All specific model classes should inherit from this class.

**Key Methods and Attributes**:

- `_build_model()`: Abstract method to build the neural network model.
- `get_output(input_tensor)`: Runs the model on the input tensor and retrieves outputs.
- `extract_layers`: List of layer names from which to extract outputs.
- `preprocess`: Function to preprocess input images.

### BaseTask

An abstract base class to represent a task in the benchmarking framework. All specific task classes should inherit from this class.

**Key Methods and Attributes**:

- `compute_task_performance(pairs_distances_df)`: Abstract method to compute the task performance metrics.
- `pairs_df`: DataFrame containing pairs of images and related information.
- `distance_metric`: Function used to compute the distance between image embeddings.

### Derived Task Classes

#### AccuracyTask

Evaluates the accuracy of the model's predictions.

- **Usage**: Used for tasks where ground truth labels are available.
- **Metrics Computed**: Accuracy, AUC, Optimal Threshold.

#### CorrelationTask

Evaluates the correlation between the model's computed distances and the true distances.

- **Usage**: Used when comparing computed distances with known distances.
- **Metrics Computed**: Correlation Score.

#### RelativeDifferenceTask

Computes the relative difference between two groups in the computed distances.

- **Usage**: Useful for bias detection or group comparison tasks.
- **Metrics Computed**: Group Means, Relative Difference.

### PlotHelper

A helper class for generating plots related to model performance on tasks.

**Key Methods**:

- `bar_plot()`: Creates and saves bar plots for task performances.
- `scatter_plot()`: Creates and saves scatter plots showing the correlation.

### MultiModelTaskManager

Manages multiple models and tasks, facilitating the computation of task performance across models.

**Key Methods**:

- `add_models(models)`: Adds models to the manager.
- `add_tasks(tasks)`: Adds tasks to the manager.
- `run_task(model_name, task_name)`: Runs a specific task on a specific model.
- `run_all_tasks_all_models()`: Runs all tasks on all models.

### PairDataset

A custom `Dataset` class that provides pairs of images for processing.

**Key Methods**:

- `__getitem__(idx)`: Retrieves the image tensors and associated metadata for the given index.

## Usage

### Prerequisites

- Python 3.7 or higher
- PyTorch
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- torchvision
- tqdm

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/omri-am/FacesBenchmark.git
   ```

2. **Install required packages**:

   ```bash
   pip install -r requirements.txt
   ```

### Defining Models

Create a class that inherits from `BaseModel` and implements the `_build_model()` and `_forward()` methods.

```python
class MyModel(BaseModel):
    def _build_model(self):
        # Build your model architecture
        self.model = ...

    def _forward(self, input_tensor):
        # Define the forward pass
        return self.model(input_tensor)
```

### Defining Tasks

Create a class that inherits from `BaseTask` and implements the `compute_task_performance()` method.

```python
class MyTask(BaseTask):
    def compute_task_performance(self, pairs_distances_df):
        # Compute performance metrics
        performance_metrics = ...
        return performance_metrics
```

### Running Tasks

Use `MultiModelTaskManager` to manage and run tasks across models.

```python
# Instantiate models and tasks
model = MyModel(name='MyModel')
task = MyTask(name='MyTask', pairs_file_path='pairs.csv', images_path='images', distance_metric=...)

# Initialize the manager
manager = MultiModelTaskManager(models=[model], tasks=[task])

# Run a specific task on a model
manager.run_task(model_name='MyModel', task_name='MyTask', print_log=True)
```

### Visualizing Results

The `PlotHelper` class provides static methods to generate plots.

```python
# Generate bar plot
PlotHelper.bar_plot(performances=performance_df, y='Accuracy', ylabel='Accuracy', ...)

# Generate scatter plot
PlotHelper.scatter_plot(performances=performance_df, distances=distances_dict, ...)
```

## Examples

### Example: Running an Accuracy Task

```python
from models import MyModel
from tasks import AccuracyTask
from manager import MultiModelTaskManager

# Define the model
model = MyModel(name='ResNet50')

# Define the task
task = AccuracyTask(
    name='FaceRecognition',
    pairs_file_path='data/pairs.csv',
    images_path='data/images',
    distance_metric=pairwise.cosine_distances
)

# Initialize the manager
manager = MultiModelTaskManager(models=[model], tasks=[task])

# Run the task
manager.run_task(model_name='ResNet50', task_name='FaceRecognition', export_path='results', print_log=True)
```

### Example: Running All Tasks on All Models

```python
from models import MyModel1, MyModel2
from tasks import AccuracyTask, CorrelationTask
from manager import MultiModelTaskManager

# Define models
model1 = MyModel1(name='Model1')
model2 = MyModel2(name='Model2')

# Define tasks
task1 = AccuracyTask(name='Task1', pairs_file_path='...', images_path='...', distance_metric=...)
task2 = CorrelationTask(name='Task2', pairs_file_path='...', images_path='...', distance_metric=...)

# Initialize the manager
manager = MultiModelTaskManager(models=[model1, model2], tasks=[task1, task2])

# Run all tasks on all models
manager.run_all_tasks_all_models(export_path='results', print_log=True)
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with your improvements.

### Steps to Contribute

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Test your changes thoroughly.
5. Submit a pull request.

# Acknowledgements

- **PyTorch**: An open-source machine learning framework.
- **pandas**: Data manipulation and analysis.
- **NumPy**: Numerical computing library.
- **scikit-learn**: Machine learning library.
- **Matplotlib & Seaborn**: Visualization libraries.

# Contact

For questions or support, please open an issue or contact:

Omri Amit, [omriamit@mail.tau.ac.il](mailto:omriamit@mail.tau.ac.il)

Mia Shlein, [miachaias@mail.tau.ac.il](mailto:miachaias@mail.tau.ac.il)
