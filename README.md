# Towards Generalizable Data-Driven Pharmacokinetics with Interpretable Neural ODEs

[![Paper](https://img.shields.io/badge/Paper-JCIM-blue)](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.5c02924)
[![App](https://img.shields.io/badge/APP-Uni--PK-green)](https://www.bohrium.com/apps/uni-pk)
[![Demo](https://img.shields.io/badge/Demo-Uni--PK-green)](https://funmg.dp.tech/uni-pk/#/)

## Introduction
This repository contains the source code for the paper **"Towards Generalizable Data-Driven Pharmacokinetics with Interpretable Neural ODEs"**, published in the *Journal of Chemical Information and Modeling (JCIM)*. 

The project aims to leverage interpretable Neural Ordinary Differential Equations (Neural ODEs) for data-driven pharmacokinetics (PK) modeling, providing a robust and generalizable framework for drug concentration prediction.

## Table of Contents
- [Model Weights](#model-weights)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
- [Citation](#citation)

---

## Model Weights 📥
Due to the large file size, pre-trained model weights are hosted on **GitHub Releases**. You must download them manually to run the inference.

1. **Download**: Go to [v1.0.0 Release](https://github.com/emotionor/UniPKModel/releases/tag/v1.0.0).
2. **Placement**: Place the downloaded weight files into the `./examples/weights/` directory.

**Verification (Optional):**
You can verify the file integrity using SHA-256:
```bash
sha256sum ./examples/weights/pk_NeuralODE_3_log_mae_time_exp_decay_128_128_24

```

---

## Installation

To set up the project, follow these steps:

1. **Clone the repository:**
```bash
git clone [https://github.com/emotionor/UniPKModel.git](https://github.com/emotionor/UniPKModel.git)
cd UniPKModel

```


2. **Install dependencies:**
```bash
pip install -r requirements.txt

```



---

## Data

The dataset used in this project is located in the `examples/data` directory. It contains 277 Rat pharmacokinetic (PK) curves, structured for training and validating Neural ODE models.

---

## Usage

### PK Inference

Ensure the model weights are correctly placed in the weights directory before execution.

```python
from train_pk import test_model

# Path to the pre-trained weights
model_path = './examples/weights/pk_NeuralODE_3_log_mae_time_exp_decay_128_128_24'

# Path to the test data
test_filepath = './examples/data/CT1127_clean_iv_test.csv'

# Run inference
test_model(model_path, test_filepath=test_filepath, output_path='./examples/outputs')

```

You can also test the model through our web-based platform: [Uni-PK](https://www.bohrium.com/apps/uni-pk)

---

## Citation

If you find this work useful in your research, please cite our paper:

**Towards Generalizable Data-Driven Pharmacokinetics with Interpretable Neural ODEs** *Journal of Chemical Information and Modeling* (2026)

DOI: [10.1021/acs.jcim.5c02924](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.5c02924)

```bibtex
@article{cui2025toward,
  title={Toward Generalizable Data-Driven Pharmacokinetics with Interpretable Neural ODEs},
  author={Cui, Yaning and Ji, Xiaohong and Guo, Wentao and Chen, Shangqian and Shen, Tao and Chen, Liye and Ke, Guolin and Jin, Chuanfei and Gao, Zhifeng and Sun, Weijie},
  journal={Journal of Chemical Information and Modeling},
  year={2025},
  publisher={ACS Publications}
  doi={10.1021/acs.jcim.5c02924}
}

```