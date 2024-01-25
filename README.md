# Stage Classification


**Computer Vision Assignment**

This project implements various deep learning algorithms for classifying various stages of construction for a given time series data of a structure at a construction site. 


## Overview

The project is developed using the lightweight Pytorch wrapper, [Lightning Modules](https://lightning.ai/docs/pytorch/stable/starter/style_guide.html) and uses system-model style to structure the code. This lets us self-contain the main skeleton of the code inside the system class and all the neural models in the model class. We call the system class `Classifier_ti.py` whereas the models are implemented with their own names as `cnn.py` and `lstm.py`.

Task 0: Data Loading

Data cleaning, pre-processing, and loading are carried out to prepare the data for training the neural networks. The routine for data loading and visualization is presented in [this jupyter notebook](run_scripts/task0_dataloading.ipynb).

Task 1: Image Classification

We use a 3-stage convolutional network followed by linear layers to implement an image classifier. The system class for this implementation can be found [here](models/classifier_t1.py), and the model used for this architecture can be found [here](models/cnn.py).

Task 2: Time-series Classification

We use two different models to perform time-series image classification. The first model is based on [LSTM networks](models/lstm.py), whereas the second model is based on [CNN and LSTM networks](models/cnn_lstm.py). The system class for this implementation can be found [here](models/classifier_t2.py).

## Training and Inference

Training info:
- Computational resources: NVIDIA V100 Tensor Core GPU
- Maximum iterations: 10k to 100k
- Time taken by each experiment: 1 to 1.5hrs

The jupyter notebooks and configuration files to train the models can be found [here](run_scripts/) and [here](run_scripts/configs/).

To run the trained models:
- Download the dataset --> Create a folder `./data` and place all the data files.
- Download the checkpoint from [here](https://syncandshare.lrz.de/getlink/fiJH2GvJvkxVoayfTQy2TR/task1_cnn.ckpt) and place it in folder [run_scripts/lightning_logs/checkpoints/](run_scripts/lightning_logs/checkpoints/).
- Run the jupyter notebook [trainedmodels-loadckpt.ipynb](run_scripts/trainedmodels-loadckpt.ipynb)


## Setting Up
### Dependencies
This module uses several packages :
- **numpy**, **matplotlib**
- **pandas**, **omegaconf**
- **torch**, **pytorch_lightning**, **torchvision**
- **sklearn**, **PIL** 


### Structure

The project is organized as follows:
- **data**: This folder contains all the data files used for training the models.
- **models** This folder contains all the NN models (CNN, LSTM, CNN+LSTM) and the system files.  
- **run_scripts** This folder contained the jupyter notebook for training and evaluating the models.
- **utilities** This folder contains files to handle data loading, pre-processing, and cleaning.



## Author

- **Akshay Pimpalkar**




