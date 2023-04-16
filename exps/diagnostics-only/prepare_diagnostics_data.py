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
    manifest_test_path:str,
    ratio:list = [0.7,0.2,0.1],
    random_seed:int = 2023
    ):

    """
    Input:
    ---
    wav_folder: path to the folder where all audio files are stored. Default as './data_og/Cambridge/wav/EN'
    metadata_path: path to the metadata file (.csv). Default as './data_og/Cambridge/metadata/EN-metadata.csv'
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
        unzip_audio_file(wav_folder, audio_archive_path)
    else: print('Found extracted audio files. Skip unzipping.')

    # List files and create manifest from list
    logger.info(
        f"Creating {manifest_train_path}, {manifest_valid_path}, and {manifest_test_path}"
    )
    
    # Creating json files for train, valid, and test all at once
    create_json(wav_folder, 
                metadata_path, 
                [manifest_train_path,manifest_valid_path,manifest_test_path],
                ratio,
                random_seed)


def create_json(wav_folder:str, metadata_path:str, manifest_paths:list, ratio:list, random_seed:int):
    """
    Creates the manifest file given the metadata file.
    """

    # Load metadata file
    df_metadata = pd.read_csv(metadata_path, sep=';')
    # Calculate total number of audio files
    wav_files = glob.glob(os.path.join(wav_folder, "*.wav"), recursive=True)
    print("Total wav audio files {} in the audio folder".format(len(wav_files)))
    # Sanity check if number of files in metadata is in consistency with number of files in the audio folder
    assert len(wav_files) == df_metadata.shape[0], "Number of audio files in the folder is not consistent with number of samples in the metadata"

    # Split the metadata file into train,valid,and test files
    df_train, df_valid, df_test = split_metadata(df_metadata, ratio, random_seed)
    dataframe_to_json(df_train,manifest_paths[0])
    dataframe_to_json(df_valid,manifest_paths[1])
    dataframe_to_json(df_test,manifest_paths[2])

    logger.info(f"{manifest_paths} successfully created!")


def split_metadata(metadata_df, ratio, random_seed):
    """
    User-independent split with each split contains similar symptomatic/non-symptomatic ratio
    """

    mask_1 = metadata_df['Symptom-label'] == 'symptomatic'
    mask_2 = metadata_df['Symptom-label'] == 'non'
    uid_pos = list(metadata_df[mask_1]['Uid'].unique())
    uid_neg = list(metadata_df[mask_2]['Uid'].unique())
    uid_pos = list(set(uid_pos).difference(set(uid_neg)))
    
    random.Random(random_seed).shuffle(uid_pos)
    random.Random(random_seed).shuffle(uid_neg)

    train_spk_pos = uid_pos[:int(len(uid_pos)*ratio[0])]
    train_spk_neg = uid_neg[:int(len(uid_neg)**ratio[0])]
    valid_spk_pos = uid_pos[int(len(uid_pos)**ratio[0]):int(len(uid_pos)*ratio[1])]
    valid_spk_neg = uid_neg[int(len(uid_neg)**ratio[0]):int(len(uid_pos)*ratio[1])]
    test_spk_pos = uid_pos[int(len(uid_pos)*ratio[1]):]
    test_spk_neg = uid_neg[int(len(uid_neg)*ratio[1]):]

    train_spk = train_spk_pos + train_spk_neg
    valid_spk = valid_spk_pos + valid_spk_neg
    test_spk = test_spk_pos + test_spk_neg

    train_df = metadata_df[metadata_df['Uid'].isin(train_spk)]
    valid_df = metadata_df[metadata_df['Uid'].isin(valid_spk)]
    test_df = metadata_df[metadata_df['Uid'].isin(test_spk)]

    return train_df, valid_df, test_df


def dataframe_to_json(df,save_path):
    # we now build JSON examples 
    examples = {}
    for _, row in df.iterrows():
        utt_id = Path(row['voice-path-new']).stem # returns the name (without extension) E.g., '00000','00010'
        examples[utt_id] = {"ID": utt_id,
                            "file_path": row['voice-path-new'], 
                            "symptom-label": row['Symptom-label'],
                            "symptom": row['Symptoms'],
                            "length": torchaudio.info(row['voice-path-new']).num_frames}
        
    if not os.path.exists(os.path.dirname(save_path)):
        Path(os.path.dirname(save_path)).mkdir(parents=True)

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
    shutil.unpack_archive(os.path.dirname(audio_archive_path), destination) # this will create a folder called 'EN' inside of the 'wav' folder
