#!/usr/bin/env python3
"""Recipe for training a symptom diagnostics model.

To run this recipe, do the following:
> python train.py manifest/train.yaml

To read the code, first scroll to the bottom to see the "main" code.
This gives a high-level overview of what is going on, while the
Brain class definition provides the details of what happens
for each batch during training.

Noise and reverberation are automatically added to each sample from OpenRIR.

Authors
 * Yi Zhu 2023
"""
import os
import sys
sys.path.append('./model')
import torch
import torchaudio
import torchaudio.functional as F
import speechbrain as sb
from hyperpyyaml import load_hyperpyyaml
from prepare_task1_data import prepare_data


# Brain class for speech enhancement training
class DiagnosticsBrain(sb.Brain):
    """Class that manages the training loop."""

    def compute_forward(self, batch, stage):
        """Runs all the computation of that transforms the input into the
        output probabilities over the N classes.
        """
        batch = batch.to(self.device)
        wavs, _ = self.augment_input(batch.signal, stage)
        # wavs = wavs.squeeze()
        predictions = self.modules.model(wavs)
        return predictions
    
    def augment_input(self, wavs, stage):
        wavs, lens = wavs
        # Add augmentation if specified. In this version of augmentation, we
        # concatenate the original and the augment batches in a single bigger batch.
        if stage == sb.Stage.TRAIN:
            if hasattr(self.hparams, "env_corrupt"):
                wavs_noise = self.hparams.env_corrupt(wavs, lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                lens = torch.cat([lens, lens])

            if hasattr(self.hparams, "augmentation"):
                wavs = self.hparams.augmentation(wavs, lens)

        return wavs, lens


    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.

        Arguments
        ---------
        predictions : tensor
            The output tensor from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """
        _, lens = batch.signal
        lab, _ = batch.symptom_label_tensor
        lab = lab.to(self.device)

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN and hasattr(self.hparams, "env_corrupt"):
            lab = torch.cat([lab, lab], dim=0)
            lens = torch.cat([lens, lens])

        # Compute the cost function: BCE for a fake/genuine classification problem
        # weight = torch.tensor([1]).to(self.device)
        loss = sb.nnet.losses.bce_loss(predictions, lab)

        # Append this batch of losses to the loss metric for easy
        self.loss_metric.append(
            batch.id, predictions, lab, lens, reduction="batch"
        )

        # Compute classification error at test time
        if stage != sb.Stage.TRAIN:
            self.error_metrics.append(batch.id, predictions, lab)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up statistics trackers for this stage
        self.loss_metric = sb.utils.metric_stats.MetricStats(
            metric=sb.nnet.losses.bce_loss
        )

        # Set up evaluation-only statistics trackers
        if stage != sb.Stage.TRAIN:
            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Store the train loss until the validation stage.
        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
        else:
            metrics = self.error_metrics.summarize() 
            stats = {
                "loss": stage_loss,
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "F1": metrics["F-score"],
            }

        # At the end of validation...
        if stage == sb.Stage.VALID:

            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats=stats,
            )

            # Save the current checkpoint and delete previous checkpoints, based on F1
            self.checkpointer.save_and_keep_only(meta=stats, max_keys=["F1"])

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats=stats,
            )


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    We expect `prepare_deepfake_trackXX` to have been called before this,
    so that the `train.csv`, `valid.csv` manifest files are available.
    """

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("file_path")
    @sb.utils.data_pipeline.provides("signal","duration")
    def audio_pipeline(file_path):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        
        signal, sr_og = torchaudio.load(file_path)
        # handle multi-channel
        if signal.shape[0] > 1:
            signal = torch.mean(signal, axis=0)

        if sr_og != 16000:
            signal = F.resample(signal,sr_og,new_freq=16000)

        signal  = signal.squeeze()
        signal = signal / torch.max(torch.abs(signal))
        duration = len(signal)
        yield signal
        yield duration
    
    # Define label pipeline:
    @sb.utils.data_pipeline.takes("symptom_label")
    @sb.utils.data_pipeline.provides("symptom_label_tensor")
    def label_pipeline(label):
        """Defines the pipeline to process the input label."""
        symptom_label_tensor = torch.tensor(label).view(1,-1)[0].type(torch.LongTensor)
        yield symptom_label_tensor

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    data_info = {
        "train": hparams["train_annotation"],
        "valid": hparams["valid_annotation"],
        "test": hparams["test_annotation"]
    }

    hparams["dataloader_options"]["shuffle"] = False

    for dataset in data_info:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=data_info[dataset],
            # replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "signal", "duration", "file_path", "symptom_label_tensor", "symptom"],
        ).filtered_sorted(sort_key="duration", reverse=False)

    # # Load or compute the label encoder (with multi-GPU DDP support)
    # # Please, take a look into the lab_enc_file to see the label to index
    # # mapping.
    # lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    # label_encoder.load_or_create(
    #     path=lab_enc_file,
    #     from_didatasets=[datasets["train"]],
    #     output_key="symptom",
    # )

    return datasets


# Recipe begins!
if __name__ == "__main__":

    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Data preparation, to be run on only one process.
    if not hparams["skip_prep"]:
        sb.utils.distributed.run_on_main(
            prepare_data,
            kwargs={
                "wav_folder": hparams["wav_folder"],
                "audio_archive_path": hparams["audio_archive_path"],
                "metadata_path": hparams["metadata_path"],
                "manifest_train_path": hparams["train_annotation"],
                "manifest_valid_path": hparams["valid_annotation"],
                "manifest_test_path": hparams["test_annotation"]
            },
        )

    # Create dataset objects
    datasets = dataio_prep(hparams)

    # Initialize the Brain object to prepare for mask training.
    brain = DiagnosticsBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # training loop
    brain.fit(
        epoch_counter=brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Load the best checkpoint for evaluation
    test_stats = brain.evaluate(
        test_set=datasets["test"],
        # min_key="error",
        test_loader_kwargs=hparams["dataloader_options"],
    )
