# AutoEval

## TableTransformer - Dataset Extractor

### Getting Started

Ensure all required libraries are installed. You can install them using pip:
```bash
python -m pip install -r requirements.txt
```

### Usage

* Performs table and cell extraction from images using pre-trained models for table detection and structure recognition. 
It includes functionalities to visualize detected tables, crop and classify cells, and create a graphical user interface (GUI) for cell classification.

* The images from the dataset are passed in the model to extraxt the last column of the table that contains the handwritten answers : "True", "False"

* Follow the GUI prompts to classify extracted cell images into categories: "true", "false", or "none"

## TinyVGG Model 

### Getting Started

```bash
pip install torch torchvision matplotlib tqdm
```

### Dataset

The dataset is organized into training and testing directories, each containing images classified into 'false', 'none', and 'true'.

### Overview 

* The TinyVGG model is a simple CNN architecture with two convolutional blocks and a fully connected classifier

* The `torchvision.transforms` module is used to apply transformations to the dataset.

* The transformed datasets are then loaded using `ImageFolder`

* The training time is measured using the `timeit` library

### Acknowledgement

The helper functions are downloaded from Daniel Bourke's PyTorch repository:
```bash
https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py
```
