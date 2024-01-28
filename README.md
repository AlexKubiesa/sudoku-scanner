# Overview

This project implements a Sudoku scanner that takes an image of a Sudoku puzzle and converts it into a digital format. The scanner has 98% accuracy on this dataset and takes about 6 seconds per puzzle.

It mainly deals with single digits per cell but also has a separate "candidates" label to cover cases where the cell is not filled but has candidates in it.

# Project structure

The `datasets` folder contains three relevant datasets:
* The `end_to_end` dataset, which captures the overall problem of taking a puzzle image and outputting the information of what is in each cell.
* The `puzzle_localization` dataset, which deals with the task of taking a puzzle image and outputting the coordinates of the puzzle corners.
* The `digit_classification` dataset, which deals with the task of taking an image of a single cell and classifying it according to which digits are in the cell.

The `end_to_end` dataset is the main one; the other two datasets are intended to help break the problem down into parts, solve these parts and then combine the solutions into an end-to-end solution.

The `models` folder contains solutions to the tasks given by the datasets.
* The `end_to_end` model solves the `end_to_end` task, using the other two models for their respective subtasks.
* The `puzzle_localizer` solves the `puzzle_localization` task using a heuristic, rule-based method.
* The `digit_classifier` solves the `digit_classification` task using a neural network trained in TensorFlow.

There are also some Jupyter notebooks in the root folder:
* `evaluate_end_to_end_model.ipynb` evaluates the performance of the end-to-end model on the end-to-end task. After making any changes to the end-to-end model, you can measure its performance on the validation set and check if it improves. The notebook can also be used to debug the model when it's not behaving as expected.
* `evaluate_puzzle_localizer.ipynb` evaluates the performance of the puzzle localizer only. This can be used to test changes to the puzzle localizer before incorporating the updated puzzle localizer into the end-to-end model.
* `train_digit_classifier.ipynb` trains a new digit classifier and evaluates its performance on the digit classification task.

# Setup

## Tested configurations

I have tested the code with the following system:
* Windows 10
* GeForce GTX 1060 6GB GPU
* Intel Core i7-7700K CPU
* 32 GB RAM (I'm sure less is fine)

## Install required packages

Make a Python virtual environment and activate it.

Then run the following on the command line.

```
python -m pip install -r requirements.txt
```

## Install CUDA and CuDNN

The best versions are CUDA 11.1.1 and cuDNN 8.1.1. This is despite the TensorFlow compatibility matrix listing CUDA 11.2 with cuDNN 8.1. If CUDA 11.2 is used, then tf.image.rgb_to_grayscale fails with some low-level GPU error.

Without CUDA and cuDNN installed, training will run on CPU which is much slower.

# Going further 

Here are some suggestions for how to improve on this solution or adapt it to your use case:
* Evaluate the end-to-end model on your own dataset. If it does badly, you could try changing the parameters of the puzzle localizer or training the digit classifier on a different dataset.
* Make the puzzle localizer faster.
* Make the digit classifier do multilabel classification so it can tell which candidates are in a cell instead of just returning "candidates".

