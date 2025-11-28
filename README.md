# Star-CDR Model Project Description

This project demonstrates the core structure and operational workflow of the Star-CDR model using synthetic data, aiming to provide a clear model implementation framework for better understanding.

## Quick Start

### 1. Environment Setup

This project uses Python version 3.13. First, install the required Python dependency packages:

```
# Install dependencies using pip

pip install -r requirements.txt
```

### 2. synthetic data Generation

Generate synthetic data that meets configuration requirements by running the script. Data will be automatically saved to the `dataset/`  directory:

```
# Execute data generation script

bash run.bash
```

Data generation rules are defined by configuration files in the `configs/` directory (such as data format, sample size, feature dimensions, etc.)

### 3. Model Training

Run the training script to start the model training process:

```
# Execute training code

python train.py
```

The training process will load synthetic data from `dataset/`


## Directory Structure

```
├── configs/          # Configuration files directory

├── dataset/          # synthetic data storage directory

├── requirements.txt  # Project dependency list

├── run.bash          # synthetic data generation script

├── train.py          # Model training main program

├── star_cdr.py       # Model structure definition

└── dataloader.py     # Data loading code

└── generate_fake_dataset.py         # synthetic data generation code

└── README.md         # Project documentation
```
