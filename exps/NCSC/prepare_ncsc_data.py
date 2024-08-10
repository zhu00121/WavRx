"""
TODO: add description
"""

import os, glob
import random
import pandas as pd
import json
from path import Path
import shutil
import logging
import torchaudio

logger = logging.getLogger(__name__)


def prepare_data(
    wav_folder:str,
    audio_archive_path:str,
    metadata_path:str,
    manifest_train_path:str,
    manifest_valid_path:str,
    manifest_test_path:str
    ):

    """
    Input:
    ---
    wav_folder: path to the folder where all audio files are stored. Default as './data_og/NCSC/wav/'
    metadata_path: path to the metadata file (.csv). Default as './data_og/NCSC/metadata/NCSC-metadata.csv'
    manifest_path: path to the manifest file (.json).
    """

    # Check if this phase is already done (if so, skip it)
    if skip(manifest_train_path, manifest_valid_path, manifest_test_path):
        logger.info("Manifest files preparation completed in previous run, skipping.")
        return
    
    if not os.path.exists(metadata_path):
        raise ValueError("Metadata file not found. Check the metadata folder. ")

    # If the wav folder does not exist, unzip the audio zip file
    if not check_folders(wav_folder):
        print('Unzipping audio zip file...')
        unzip_audio_file(wav_folder, audio_archive_path)
    else: print('Found extracted audio files. Skip unzipping.')

    # List files and create manifest from list
    logger.info(
        f"Creating {manifest_train_path}, {manifest_valid_path}, and {manifest_test_path}"
    )
    
    # Creating json files for train, valid, and test all at once
    create_json(wav_folder, 
                metadata_path, 
                [manifest_train_path,manifest_valid_path,manifest_test_path]
                )


def create_json(wav_folder:str, metadata_path:str, manifest_paths:list):
    """
    Creates the manifest file given the metadata file.
    """
    # Load metadata file
    df_metadata = pd.read_csv(metadata_path, sep=';')

    # Split the metadata file into train,valid,and test files
    df_train = df_metadata[df_metadata['split']==0]
    dataframe_to_json(df_train,manifest_paths[0])
    df_valid = df_metadata[df_metadata['split']==1]
    dataframe_to_json(df_valid,manifest_paths[1])
    df_test = df_metadata[df_metadata['split']==2]
    dataframe_to_json(df_test,manifest_paths[2])

    logger.info(f"{manifest_paths} successfully created!")


def dataframe_to_json(df,save_path):
    # we now build JSON examples 
    examples = {}
    for _, row in df.iterrows():
        utt_id = row['Uid']
        examples[utt_id] = {"ID": utt_id,
                            "file_path": row['voice-path-new'], 
                            "intl_label": row['label']}
        
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))

    with open(save_path, "w") as f:
        json.dump(examples, f, indent=4)

    return examples


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


def unzip_audio_file(destination, audio_archive_path):
    """
    Unzip the compressed audio folder.
    """
    if not os.path.exists(audio_archive_path):
        raise ValueError("Audio zip file not found. Please refer to prep.ipynb first to prepare the zip file.")
    shutil.unpack_archive(audio_archive_path, os.path.dirname(destination))
