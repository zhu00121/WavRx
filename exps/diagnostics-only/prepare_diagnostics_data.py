"""
TODO: add description
"""

import os, glob
import pandas as pd
import json
from path import Path
import shutil
import logging
import torchaudio

logger = logging.getLogger(__name__)


def prepare_diagnostics_data(
    wav_folder:str,
    metadata_path:str,
    manifest_path:str
    ):

    """
    Input:
    ---
    wav_folder: path to the folder where all audio files are stored. Default as './data_og/Cambridge/wav/EN'
    metadata_path: path to the metadata file (.csv). Default as './data_og/Cambridge/metadata/EN-metadata.csv'
    manifest_path: path to the manifest file (.json).
    """

    # Check if this phase is already done (if so, skip it)
    if skip(manifest_path):
        logger.info("Manifest file preparation completed in previous run, skipping.")
        return
    
    if not os.path.exists(metadata_path):
        raise ValueError("Metadata file not found. Check the metadata folder. ")

    # If the wav folder does not exist, unzip the audio zip file
    if not check_folders(wav_folder): 
        unzip_audio_file(os.path.dirname(wav_folder))

    # List files and create manifest from list
    logger.info(
        f"Creating {manifest_path}"
    )
    
    # Creating json files
    create_json(wav_folder, metadata_path, manifest_path)


def create_json(wav_folder:str, metadata_path:str, manifest_path:str):
    """
    Creates the manifest file given the metadata file.
    """

    # Load metadata file
    df_metadata = pd.read_csv(metadata_path, header=True, sep=';')
    # Calculate total number of audio files
    wav_files = glob.glob(os.path.join(wav_folder, "/*.wav"), recursive=True)
    print("Total wav audio files {} in the audio folder".format(len(wav_files)))

    # Sanity check if number of files in metadata is in consistency with number of files in the audio folder
    assert len(wav_files) == df_metadata.shape[0], "Number of audio files in the folder is not consistent with number of samples in the metadata"

    # we now build JSON examples 
    examples = {}
    for _, row in df_metadata.iterrows():
        utt_id = Path(row['voice-path-new']).stem # returns the name (without extension) E.g., '00000','00010'
        examples[utt_id] = {"ID": utt_id,
                            "file_path": row['voice-path-new'], 
                            "symptom-label": row['Symptom-label'],
                            "symptom": row['Symptoms'],
                            "length": torchaudio.info(row['voice-path-new']).num_frames}

    with open(manifest_path, "w") as f:
        json.dump(examples, f, indent=4)

    logger.info(f"{manifest_path} successfully created!")


def skip(*filenames):
    """
    Detects if the data preparation has been already done.
    If the preparation has been done, we can skip it.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """
    for filename in filenames:
        if not os.path.isfile(filename):
            return False
    return True


def check_folders(*folders):
    """Returns False if any passed folder does not exist."""
    for folder in folders:
        if not os.path.exists(folder):
            return False
    return True


def unzip_audio_file(destination):
    """
    Unzip the compressed audio folder.
    """
    audio_archive = os.path.join(destination, "EN.zip")
    if not os.path.exists(audio_archive):
        raise ValueError("Audio zip file not found. Please refer to prep.ipynb first to prepare the zip file.")
    shutil.unpack_archive(audio_archive, destination) # this will create a folder called 'EN' inside of the 'wav' folder
