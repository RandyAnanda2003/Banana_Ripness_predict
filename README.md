# Banana Ripening Process Detection

This repository contains code for detecting the ripeness stage of bananas using an object detection model. The model can identify different stages such as fresh ripe, fresh unripe, overripe, ripe, rotten, and unripe bananas.

## Requirements

Ensure you have the following installed:

- Python 3.7 or higher
- pip

## Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
    ```

2. **Install Required Libraries**

    ```bash
    pip install -r requirements.txt
    ```

    If you don't have `requirements.txt`, create one with the following content:

    ```text
    tensorflow
    numpy
    opencv-python
    ```

3. **Download the Dataset**

    Download the dataset from RoboFlow using the following link: [Banana Ripening Process Dataset](https://universe.roboflow.com/fruit-ripening/banana-ripening-process/dataset/2)

4. **Extract the Dataset**

    After downloading, extract the dataset to a directory of your choice. Update the dataset path in the script if necessary.

## Usage

1. **Prepare the Dataset**

    Ensure the dataset is extracted and accessible by the script. Update the dataset path in `model2.py` and `model4.py` if required.

2. **Run the Model**

    Use the provided Python scripts to run the model. Here is an example:

    ```bash
    python model2.py
    ```

    or

    ```bash
    python model4.py
    ```

    Ensure to replace the dataset path in the script if necessary.

## Scripts

- `model2.py`: Script for training and evaluating the model.
- `model4.py`: Alternative script for model training and evaluation.

## Dataset Information

The dataset contains images of bananas at different ripening stages with the following classes:

- freshripe
- freshunripe
- overripe
- ripe
- rotten
- unripe

For more details, refer to the [dataset information](README.dataset.txt).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This project uses the Banana Ripening Process Dataset provided by [RoboFlow](https://public.roboflow.ai/object-detection/undefined) under the CC BY 4.0 license.
