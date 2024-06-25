# AutoEval

## TableTransformer - Dataset Extractor

### Getting Started
- First install pytorch with cuda from the [pytorch](https://pytorch.org/get-started/locally/) website.

- Ensure all required libraries are installed. You can install them using pip:
```bash
pip install -r requirements.txt
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

## Evaluation

### Imports and Setup

Required libraries are imported, and the device is configured (CUDA or CPU).

### Function Definitions

* `MaxResize`: Resizes images while maintaining aspect ratio.
* `box_cxcywh_to_xyxy and rescale_bboxes`: For bounding box processing.
* `outputs_to_objects`: Converts model outputs to objects.
* `fig2img`: Converts a Matplotlib figure to a PIL Image.
* `visualize_detected_tables`: Visualizes detected tables with bounding boxes.
* `objects_to_crops`: Processes bounding boxes into cropped images and tokens.
* `get_cell_coordinates_by_row`: Extracts cell coordinates by row.

### Model Configuration

Loads pre-trained models for table detection and structure recognition.

### Model Definition

TinyVGG: A neural network model used for classification of cell contents.
The model is instantiated and loaded with pre-trained weights.

### Dataset Preparation

Reads correct answers from `ModelAnswer.csv` and sets up the evaluation dataset folder.

### Cell Extraction and Classification

`perform_extraction_and_classify_image` : Main function to extract table cells and classify their contents.

### Runner Code

Iterates through images, performs extraction and classification, and saves results to `marks.csv`.

