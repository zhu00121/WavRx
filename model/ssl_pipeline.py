import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2
from speechbrain.lobes.models.huggingface_whisper import HuggingFaceWhisper

# model_hub_w2v2 = "facebook/wav2vec2-base-960h"
# model_hub_whisper = "openai/whisper-tiny"

class SSL_diagnoser(nn.Module):

    def __init__(self,
                 encoder_choice:str='wav2vec2',
                 ssl_encoder_source:str=None,
                 freeze_encoder:bool=True,
                 num_input_features:int=768,
                 num_fc_neurons:int=-1,
                 num_classes:int=1,
                 dp:int=0.2,
                 quantizer:str=None):
        
        super().__init__()

        assert encoder_choice in ['wav2vec2', 'whisper', 'hubert'], "Unknown encoder choice"

        # Encoder module
        # TODO: add hubert
        if encoder_choice == 'wav2vec2':
            self.enc = HuggingFaceWav2Vec2(ssl_encoder_source, save_path='/pretrained_models')
        elif encoder_choice == 'whisper':
            self.enc = HuggingFaceWhisper(ssl_encoder_source, save_path='/pretrained_models')
        
        for param in self.enc.parameters():
            param.requires_grad = not freeze_encoder

        # Classifier module
        if num_fc_neurons == -1: num_fc_neurons = num_input_features
        self.fc = nn.Sequential(
            nn.Linear(num_input_features, num_fc_neurons),
            nn.Dropout(p=dp),
            nn.LeakyReLU(0.1),
            nn.Linear(num_fc_neurons, num_classes)
        )

        self.pooling = lambda x: F.adaptive_avg_pool1d(x, 1) 

        # TODO: add quantizer module (discrete and soft)
        self.quantizer = quantizer

    def forward(self,x):
        ssl_output = self.enc(x) # (batch, time, features)
        if self.quantizer is not None:
            ssl_output = self.quantizer(ssl_output) # (batch, time, features)
        ssl_output = self.pooling(ssl_output).squeeze(-1) # pool over time axis -> (batch, features)
        final_output = self.fc(ssl_output) # (batch, num_classes)
        return final_output
        