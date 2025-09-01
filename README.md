# Towards Generalizable Data-Driven Pharmacokinetics with Interpretable Neural ODEs

## Introduction
This repository contains the source code for the paper "Towards Generalizable Data-Driven Pharmacokinetics with Interpretable Neural ODEs." The project aims to leverage interpretable Neural Ordinary Differential Equations (Neural ODEs) for data-driven pharmacokinetics modeling.

## Table of Contents
- [Introduction](#introduction)
- [Table of Contents](#table-of-contents)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)

## Installation
To set up the project, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/emotionor/UniPKModel.git
   cd UniPKModel
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Data
The dataset used in this project is located in the `examples` directory. It contains 277 Rat pharmacokinetic (PK) curves.

## Usage
Steps to train, validate, and infer using the model:

1. **PK Inference**:

   ```python
   from train_pk import test_model

   model_path = f'./examples/weights/pk_NeuralODE_3_log_mae_time_exp_decay_128_128_24'

   test_filepath = f'./examples/data/CT1127_clean_iv_test.csv'

   test_model(model_path, test_filepath=test_filepath, output_path='./examples/outputs')
   ```

   You can also test the model using the demo website: [Uni-PK Demo](https://funmg.dp.tech/uni-pk)




