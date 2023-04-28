import torch
import torch.nn as nn
import torch.nn.functional as F
from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2
from speechbrain.lobes.models.huggingface_whisper import HuggingFaceWhisper
from transformers import AutoModel, AutoProcessor
from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling

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
            self.enc = HuggingFaceWav2Vec2(ssl_encoder_source, save_path='./pretrained_models')
        elif encoder_choice == 'whisper':
            self.enc = HuggingFaceWhisper(ssl_encoder_source, save_path='./pretrained_models')
        
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
        ssl_output = ssl_output.permute(0,2,1)
        ssl_output = self.pooling(ssl_output).squeeze(-1) # pool over time axis -> (batch, features)
        final_output = self.fc(ssl_output) # (batch, num_classes)
        return final_output



class SSL_diagnoser_v2(nn.Module):
    def __init__(
        self,
        encoder: str = "microsoft/wavlm-base-plus",
        pooling: str = "avg",
        num_features: int = 768,
        num_classes: int = 7,
        freeze_encoder: bool = True,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.feature_extractor = AutoModel.from_pretrained(encoder)
        for param in self.feature_extractor.parameters():
            param.requires_grad = not freeze_encoder

        self.fc = nn.Sequential(
            nn.Linear(num_features, num_features),
            nn.Dropout(p=0.25),
            nn.LeakyReLU(0.1),
            nn.Linear(num_features, num_classes)
        )

        self.weights_stack = nn.Parameter(torch.ones(self.feature_extractor.config.num_hidden_layers))
        try:
            self.processor = AutoProcessor.from_pretrained(encoder)
        except:
            self.processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")

        if pooling == "avg":
            self.pooling = lambda x: F.adaptive_avg_pool1d(x, 1)
        else:
            self.pooling = AttentiveStatisticsPooling(
                num_features,
                attention_channels=num_features,
                global_context=True
            )

    def forward(self, x):
        # Preprocessing the data
        input_values = self.processor(x, sampling_rate=16000, return_tensors="pt").input_values[0]
        input_values = input_values.to(device=x.device, dtype=x.dtype)

        # Extract wav2vec2 hidden states and perform a weighted sum
        features = self.feature_extractor(input_values, output_hidden_states=True)
        features = self.weighted_sum(features.hidden_states[1:])

        # Pooling (on time dimension)
        features = features.permute(0, 2, 1) # (batch, features, time) => (batch, time, features)
        features = self.pooling(features).squeeze(-1)
        output = self.fc(features)

        return output

    def weighted_sum(self, features):
        layer_num = len(features)

        # Perform the weighted sum
        stacked_feature = torch.stack(features, dim=0)

        _, *origin_shape = stacked_feature.shape
        stacked_feature = stacked_feature.view(layer_num, -1)
        norm_weights = F.softmax(self.weights_stack, dim=-1)
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature).sum(dim=0)
        weighted_feature = weighted_feature.view(*origin_shape)

        return weighted_feature