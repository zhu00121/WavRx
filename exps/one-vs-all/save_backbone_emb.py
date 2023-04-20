
import torch
import torch.nn as nn
import speechbrain as sb
from speechbrain.pretrained import EncoderClassifier
import torchaudio.functional as F
import torchaudio
import pandas as pd
from tqdm import tqdm

def read_audio(file_path):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""
        
        signal, sr_og = torchaudio.load(file_path)
        # handle multi-channel
        if signal.shape[0] > 1:
            signal = torch.mean(signal, axis=0)

        if sr_og != 16000:
            signal = F.resample(signal,sr_og,new_freq=16000,
                                lowpass_filter_width=64,
                                rolloff=0.9475937167399596,
                                resampling_method="sinc_interp_kaiser",
                                beta=14.769656459379492
                                )
        signal  = signal.squeeze()
        duration = len(signal)
        return signal, duration


def encode_and_save(backbone_choice, pt_source, metadata_csv:str, save_path:str):
    
    assert backbone_choice in ['ECAPA', 'XVECTOR']
    if backbone_choice == 'ECAPA':
        enc = EncoderClassifier.from_hparams(source=pt_source) # remove ''run_opts={"device":"cuda"}'' to enable generating embeddings on cpu
    elif backbone_choice == 'XVECTOR':
        enc = EncoderClassifier.from_hparams(source=pt_source)
    
    df_new = pd.DataFrame({'emb':[], 'label':[]})
    df = pd.read_csv(metadata_csv, sep=';')
    for i, row in tqdm(df.iterrows()):
        filepath = row['voice-path-new']
        label = row['Symptom-label']
        input, _ = read_audio(filepath)
        backbone_output = enc.encode_batch(wavs=input)
        backbone_output = torch.squeeze(backbone_output).numpy()
        df_new.iloc[i, 0] = backbone_output
        df_new.iloc[i, 1] = label

    # save output
    df_new.to_csv(save_path,index=False,header=True,sep=';')


def main():
     
    backbone_choice = 'ECAPA'
    pt_source = "speechbrain/spkrec-ecapa-voxceleb"
    metadata_csv = "./data_og/Cambridge/metadata/EN-metadata.csv"
    save_path = "./exps/one-vs-all/%s_emb.csv" % (backbone_choice)
    encode_and_save(backbone_choice, pt_source, metadata_csv, save_path)

if __name__ == "__main__":
    
    main()