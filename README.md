# Athena

### Inroduction
This is the repository containing all source code and results for my bachelor's thesis with the title: ATHENA: Advanced Learning through Efficient Transparent Model Harnessing with Explainable Data.

### Purpose
The purpose of this project is to repurpose explainable AI not just to explain model decisions but to enhance model efficiency by integrating the ideas of XAI into model training. The presented approach is to integrate Layer-wise Relevance Propagation (LRP) into the training of fully connected and convolutional neural networks.

### Key Features
* Standard model training
* LRP-integrated model training
* Visualization of LRP results
* Experiments for comparing the performance of standard and LRP-enhanced training

## Project Overview

* [src](src): Contains all source code for training models
    * model: Contains model code
    * rules: Contains lrp rules
    * dataloader: Responsible for retrieving MNSIT and CIFAR-10 

* [experiments](src): Contains all experiments to evaluate the performance of LRP-based approaches
    * experiment1: compares the performance between LRP and standard training
    * experiment2: compares the performance between LRP and standard training together with Early Stopping



## Usage

The easiest way to get this project running is by setting up a Python virtual environment. Follow these steps:

### Step 1: Create a Virtual Environment

To create a virtual environment, run the following command in your terminal:
```bash
python3.8 -m venv myvenv
```
Ensure that Python 3.8 is installed and used for this project. While this code was developed and tested with Python 3.8, it is likely to work with other Python versions as well.

### Step 2: Activate the Virtual Environment

Activate the virtual environment using the appropriate command for your operating system:
- On Linux/MacOS:
```bash
source myvenv/bin/activate
```
- On Windows:
```bash
myvenv\Scripts\activate
```



### Step 3: Clone the Repository

Clone this project into your local repository using the following command:
```bash
git clone <repository_url>
```
Replace `<repository_url>` with the actual URL of the repository.

### Step 4: Install the Required Python Packages

Install all necessary requirements by running:
```bash
pip install -r requirements.txt
```

### Step 5: Run an Experiment

To execute one of the experiments, navigate to the project directory and use the command line to run a specific experiment. For example:
```bash
python experiments/experiment1.py
```

This will execute the specified experiment and display the results.



