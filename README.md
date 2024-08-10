### WavRx: a generic speech health diagnostic model

This repository contains code to implement our model WavRx. The current repo only contains the model architecture, we will make the dataset set-up and training scripts available soon.

<!-- <p align="center">
  <img src="WavRx_logo.png" alt="WavRx logo" width=200/>
</p> -->

# WavRx - a speech health diagnostic model

This repository provides scripts for running a new SOTA speech health diagnostic model *WavRx*. *WavRx* obtains SOTA performance on 6 datasets covering 4 different pathologies, and shows good zero-shot generalizability. The health embeddings encoded by *WavRx* are shown to carry minimal speaker identity attributes.

This repository can be used to (1) conduct training of *WavRx* on the 6 datasets; (2) run inference using the pretrained *WavRx* backbones; (3) train and test your self-customized models on the 6 datasets without efforts needed for editing training/evaluation scripts.

For detailed information, refer to [paper](https://arxiv.org/abs/2406.18731):

```bibtex
@article{zhu2024wavrx,
  title={WavRx: a Disease-Agnostic, Generalizable, and Privacy-Preserving Speech Health Diagnostic Model},
  author={Zhu, Yi and Falk, Tiago},
  journal={arXiv preprint arXiv:2406.18731},
  year={2024}
}
```

# Table of Contents

- [Table of Contents](#table-of-contents)
- [Dependencies](#-dependencies)
- [Pretrained model](#-pretrained-model)
- [Datasets and Recipes](#-Datasets-and-Recipes)
- [Quickstart](#-quickstart)
  - [Running a single task](#Running-a-single-task)
  - [Running multiple tasks](#Runnin-multiple-tasks)
- [Train and test your own model](#-Train-and-test-your-own-model)
- [Results](#-results)
- [Contact](#-contact)
- [Citing](#-citing)

# 🛠️ Dependencies

We use *PyTorch* and *SpeechBrain* as the main frameworks. To set up the environment for *WavRx*, follow these steps:


1. Clone the repository:
   ```shell
   git clone https://github.com/zhu00121/WavRx
   cd WavRx
   ```
2. Install dependencies:
    ```
    pip install -r requirements.txt
    pip install -e .
    ```
   These commands will install the dependencies for using WavRx. 

# 🌟 Pretrained Model Backbones (to be released on HG)
| **Model**                                                                 | **Dataset**                                                                                       | **Repo**                                                         |
|--------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| WavRx-respiratory                      | [Cambridge COVID-19 Sound]()                                                                                     | [huggingface.co/](https://huggingface.co/)  |
| WavRx-COVID                       | [DiCOVA2]()                                                                                     | [huggingface.co/](https://huggingface.co/)  |
| WavRx-dysarthria                    | [TORGO]()                                                                                     | [huggingface.co/](https://huggingface.co/)  |
| WavRx-dysarthria                              | [Nemours]()                                                    | [huggingface.co/](https://huggingface.co/)                    |
| WavRx-cancer                                  | [NCSC]()                                                   | [huggingface.co/](https://huggingface.co/)     |
|

# 👷 Model Training Recipes

| **Dataset**                              | **Task**                             | **Link to recipes**                                                                       |
|------------------------------------------|--------------------------------------|----------------------------------------------------------------------------------------|
| Cambridge-Task1 | Respiraty Symptom Detection  | [link]()                                               |
| Cambridge-EN                         | Respiraty Symptom                   | [link]()|
| DiCOVA2                                | COVID-19  | [link]()|                  
| TORGO                                  | Dysarthria   | [link]()|
| Nemours                          | Dysarthria | [link]()|
| NCSC                                   | Cervical Cancer  | [link]()|

# ▶️ Quickstart

## Running a single task

## Running multiple tasks


# 📧 Contact

For questions or inquiries, you can reach the author Yi Zhu at ([yi.zhu@inrs.ca](mailto:yi.zhu@inrs.ca)).
<!-- ############################################################################################################### -->
# 📖 Citing

If you use *WavRx* and/or its backbones and/or tge training recipes, please cite:

```bibtex
@article{zhu2024wavrx,
  title={WavRx: a Disease-Agnostic, Generalizable, and Privacy-Preserving Speech Health Diagnostic Model},
  author={Zhu, Yi and Falk, Tiago},
  journal={arXiv preprint arXiv:2406.18731},
  year={2024}
}
```
