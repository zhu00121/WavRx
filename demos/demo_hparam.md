## Step-by-step intro on hparam file


This notebook is meant to walk through the structure of the hyperparameter file, and to show how it is passing the different parameters into the training script ``train.py``. 

### Locate the hparam file

First thing to do is to locate the hyperparameter file for a given dataset. They are already placed in the following path: ``exps/<DATASET>/hparam/wavrx_<DATASET>.yaml``

### Part-1: Data paths

Let's see the first part of the hparam file:

```
# shared variables
task: Cambridge_Respiratory_Task1
dataset: Cambridge
encoder_name: wavrx
freeze_encoder: True
data_folder: ./data_og

# original audio and label file paths
wav_folder: !ref <data_folder>/<dataset>/wav/TASK1-VOICE
audio_archive_path: !ref <data_folder>/<dataset>/wav/TASK1-VOICE.zip
metadata_path: !ref <data_folder>/<dataset>/metadata/TASK1-metadata.csv
```

The first few lines defined the `dataset` and `task`, we can see that they are repetitively referenced in the paths below (e.g., `wav_folder`). They need to be edited each time a new model/dataset is tested. The `encoder_name` is default as `wavrx`, this is not referenced anywhere but just for us to keep track of different models used in the experiment. The `data_folder` does not need to be changed unless you change the default folder name to something else.

The following three paths `wav_folder`, `audio_archive_path`, and `metadata_path` need to be modified each time a new dataset is tested. They tell the model where to find the audio data and the label files.

### Part-2: Data i/o

Next we see where the annotation files, training logs, and other output files are saved.
```
# Path where data manifest files will be stored
# The data manifest files are created by the data preparation script.
train_annotation: !ref ./exps/<task>/manifest/train.json
valid_annotation: !ref ./exps/<task>/manifest/valid.json
test_annotation: !ref ./exps/<task>/manifest/test.json
skip_prep: False

# data i/o paths
output_folder: !ref ./exps/<task>/brain-logs
save_folder: !ref <output_folder>/save
train_log: !ref <output_folder>/train_log.txt

# The train logger writes training statistics to a file, as well as stdout.
train_logger: !new:speechbrain.utils.train_logger.FileTrainLogger
    save_file: !ref <train_log>

ckpt_interval_minutes: 15 # save checkpoint every 15 min
```
The first three lines define where the *annotation files* are saved. These files are generated automatically by `prepare_data.py`, which converts the raw metadata file (usually `.csv` format) to `.json`. These paths do not need to be edited. The next three lines define where the training and evaluation results are saved to. By default, a `brain-logs` folder will be created in the corresponding `task` folder, which contains the training log (e.g., training loss, validation metrics, test results, etc.) and model checkpoints (default name will be `model.ckpt`). If you want to run evaluation of multiple different models for the same dataset, and save all the outputs, you can edit the path of `output_folder` to something like `exps/<task>/<encoder_name>/brain-logs`. The `encoder_name` is defined in **Part-1**.

### Part-3: Training parameters
This part of the hparam file shows the values of training hyperparameters.
```
clamp_length: 160000 # limit duration within 10s

# Training Parameters
sample_rate: 16000
number_of_epochs: 30
batch_size: 1
lr_start: 0.0001
lr_final: 0.00001
n_classes: 1
dataloader_options:
    batch_size: !ref <batch_size>
    drop_last: True
```
`clamp_length` is referenced in the `train.py` file which limits the recording duration for training efficiency. As can be seen from the `sample_rate`, we use a sampling rate of 16000, so the maximal duration of a training sample will be no longer than 10s. Other hyperparameters can be understood by their names. These parameters were kept the same across all datasets when training *WavRx*, but feel free to change them for your own model.

### Part-4: Model architecture
Here we define the parameters of *WavRx*. If you added your own model script to the `WavRx/model` folder, you can specify its parameters here.

```
model: !new:wavrx.WavRx
    ssl_encoder_source: "microsoft/wavlm-base-plus"
    num_ssl_feat: 768
    num_fc_neurons: 768
    num_classes: 1
    freeze_encoder: True
    pooling_1: 'atn'
    pooling_2: 'atn'
    dp: 0.25
    sample_rate: 50
    win_length: 256
    hop_length: 64

```
`!new:wavrx.WavRx` is pointing to the `WavRx` model class (`nn.Module`) defined in the model script. The following parameters are default as the optimal ones used for achieving the results reported in our paper. They remain the same across all datasets.

### Part-5: Others
The rest of the hparam file include other details of the training setup, such as the optimizer, epoch counter, etc. There are other options from **SpeechBrain** for optimizers and schedulers. The ones shown below are what was used for *WavRx*, but can be modified to satisfy your own needs.

```
# The first object passed to the Brain class is this "Epoch Counter"
# which is saved by the Checkpointer so that training can be resumed
# if it gets interrupted at any point.
epoch_counter: !new:speechbrain.utils.epoch_loop.EpochCounterWithStopper
    limit: !ref <number_of_epochs>
    limit_to_stop: 3
    limit_warmup: 2
    direction: 'max'

# This optimizer will be constructed by the Brain class after all parameters
# are moved to the correct device. Then it will be added to the checkpointer.
opt_class: !name:torch.optim.AdamW
    lr: !ref <lr_start>

# This function manages learning rate annealing over the epochs.
# We here use the simple lr annealing method that linearly decreases
# the lr from the initial value to the final one.
lr_annealing: !new:speechbrain.nnet.schedulers.LinearScheduler
    initial_value: !ref <lr_start>
    final_value: !ref <lr_final>
    epoch_count: !ref <number_of_epochs>

# This object is used for saving the state of training both so that it
# can be resumed if it gets interrupted, and also so that the best checkpoint
# can be later loaded for evaluation or inference.
checkpointer: !new:speechbrain.utils.checkpoints.Checkpointer
    checkpoints_dir: !ref <save_folder>
    recoverables:
        model: !ref <model>
        counter: !ref <epoch_counter>
```