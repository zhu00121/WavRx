<!-- <p align="center">
  <img src="WavRx_logo.png" alt="WavRx logo" width=200/>
</p> -->

# WavRx - a Speech Health Diagnostic Model

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
2. Create a virtual env for the repo
   ```
   python3.10.13 -m <NAME_YOUR_VENV>
   source <NAME_YOUR_VENV>/bin/activate
   ```
3. Install dependencies:
    ```
    pip install -r requirements.txt
    ```

# 🌟 Pretrained Model Backbones (to be released on HG)
Note that some employed datasets are subject to confidentiality agreement, this restriction may also apply to the pretrained model weights. We are currently working on making the pretrained backbones open-source on HuggingFace.
| **Model**                                                                 | **Dataset**                                                                                       | **Repo**                                                         |
|--------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|------------------------------------------------------------------|
| WavRx-respiratory                      | [Cambridge COVID-19 Sound]()                                                                                     | [huggingface.co/](https://huggingface.co/)  |
| WavRx-COVID                       | [DiCOVA2]()                                                                                     | [huggingface.co/](https://huggingface.co/)  |
| WavRx-dysarthria                    | [TORGO]()                                                                                     | [huggingface.co/](https://huggingface.co/)  |
| WavRx-dysarthria                              | [Nemours]()                                                    | [huggingface.co/](https://huggingface.co/)                    |
| WavRx-cancer                                  | [NCSC]()                                                   | [huggingface.co/](https://huggingface.co/)     |

# 👷 Model Training Recipes

| **Dataset**                              | **Task**                             | **Link to training recipes** | **Link to data preparation scripts**                                                                       |
|------------------------------------------|--------------------------------------|----------------------------------------------------------------------------------------|--------------------------------------|
| Cambridge-Task1 | Respiraty Symptom Detection  | [link]()                                               |[link]()|
| Cambridge-EN                         | Respiraty Symptom                   | [link]()|[link]()|
| DiCOVA2                                | COVID-19  | [link]()| [link]() |                  
| TORGO                                  | Dysarthria   | [link]()| [link]() |
| Nemours                          | Dysarthria | [link]()| [link]()|
| NCSC                                   | Cervical Cancer  | [link]()| [link]() |

# ▶️ Quickstart

## Running a single task
Since each dataset has a different dataset structure with the corresponding partition, the training receipes are therefore stored separately in different folders. *Links* in the table above can be used to locate the corresponding recipes for a given dataset. 

The steps for training *WavRx* (or your own model) are as follows:

1. Medical data are hard to access and they typically do not have similar data structures. We make this easy for you. Use the **Link to data preparation scripts** to see where to download the data files, and how to prepare each dataset in the required format.

2. [Optional only if you want to train your own model] Place the code of your model in the ``model`` folder, it needs to have 1 output neuron (without sigmoid).

3. Check the hyperparameter file at ``exps/<DATASET>/hparams/wavrx_<DATASET>.yaml``. Modify the variables if needed. We provide a detailed guidance in ``demos/demo_hparam.md`` where we walk through the hyperparam file and demonstrate how to modify it for your own usage. If you simply want to replicate our results, there is no need to change it.

4. The ``train.py`` does NOT need to be edited. Unless you want to change the training strategy or the loss function (i.e., Supervised training with BCEwithlogits loss). All the hyperparameters and the input models are controlled by modifying the hyperparam file. This helps to ensure that models are compared in a fair manner.

5. Initiate training by calling ``python train.py hparams/wavrx_<DATASET>.yaml``. The test evaluation will be automatically conducted at the end of training using the best checkpoint with the highest F1 score. The results will be automatically saved.

6. Repeat step-5 for each dataset, results will be saved independently in the corresponding `exp/<DATASET>` folder.

😎 Voila! Enjoy your model training 😎

## Running multiple tasks in one-shot
Currently the repository is not built for running multiple tasks in one-shot, while this can be done by wrapping them in one shell script. However, such function will be made available with our ongoing health benchmark project - a larger health benchmark for easy implementation and evaluation of SOTA diagnostic models for 10+ diseases. If you are interested, keep an eye on the [SpeechBrain Benchmark](https://github.com/speechbrain/benchmarks) where we will be releasing our scripts.

# 📧 Contact

For questions or inquiries, feel free to open an issue or you can reach the author Yi Zhu at ([yi.zhu@inrs.ca](mailto:yi.zhu@inrs.ca)).
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
