import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoFeatureExtractor, WavLMModel
from speechbrain.lobes.models.ECAPA_TDNN import AttentiveStatisticsPooling
from speechbrain.processing.features import STFT


#############################################
class modulation_block(nn.Module):

    """
    The modulation block can be used independently. It takes any 2-D representation with a feature and time dimension, then 
    converts it to the frequency domain. The conversion itself is a standard Short-Time Fourier Transform (STFT), as is typically
    used to convert speech waveform into spectrogram. The intuition is that SSL representations has the time-varying nature, and 
    the modulation block can extract the modulation dynamics of the temporal variations, like respiration and articulation. 
    """

    def __init__(self,
                 sr,
                 win_len,
                 hop_len,
                 keep_temporal_dim=False,
                 ):
        """
        Input:
            sr: int
                sampling rate (in Hz).
            win_len: int
                window length of the STFT (in ms).
            hop_len: int
                hop length of the STFT (in ms).
            keep_temporal_dim: boolean
                When set to False, the modulation dynamics are average over time.
        
        Examples
        --------
        >>> x = torch.randn(8,100,768) # {BATCH, TIME, FEATURE}
        >>> mb = modulation_block(50, 256, 64)
        >>> x_mod = mb(x) # {BATCH, MODULATION_FREQUENCY, FEATURE}
        """
        super(modulation_block, self).__init__()
        self.compute_STFT = STFT(sample_rate=sr, win_length=win_len, hop_length=hop_len)
        self.eps = 1e-10
        self.keep_temporal_dim = keep_temporal_dim

    def forward(self, x):
        assert x.ndim == 3, "input to the modulation block needs to be 3D (batch, time, feat)"
        x_mod = torch.abs(self.compute_STFT(x)) # (num_bacth, time, num_mod_freq, 2, num_features)
        x_mod = torch.log(x_mod.pow(2).sum(-2)+self.eps) # take log of power and combine real and imaginary parts
        if not self.keep_temporal_dim:
            x_mod_ave = torch.mean(x_mod,axis=1) # average over time axis -> (num_batch, num_mod_freq, num_features)
            return x_mod_ave
        else:
            return x_mod
        
##############################################

class WavRx(nn.Module):
    """
    WavRx is a model architecture that outperforms SSL representations on six speech diagnostic tasks, including different respiratory and neurological diseases.
    It takes WavLM as the upstream encoder, and mixes the output representation with its modulation dynamics. The learned embeddings contain maximal health information 
    while entail minimal speaker identity information.
    """
    def __init__(
        self,
        encoder_choice: str = 'wavlm',
        ssl_encoder_source: str = "microsoft/wavlm-base-plus",
        num_ssl_feat: int = 768,
        num_fc_neurons:int = 768,
        num_classes: int = 1,
        freeze_encoder: bool = True,
        pooling_1: str = 'atn',
        pooling_2: str = 'atn',
        sample_rate: int = 50,
        win_length: int = 256,
        hop_length: int = 64,
        dp = 0.25,
        *args,
        **kwargs
        ):
        """
        Input:
            encoder_choice: str
                Upstream encoder. Default as 'wavlm', but can be switched to other SSL representations, such as Wav2vec, Hubert, data2vec, etc.
            ssl_encoder_source: str
                The backbone of uptream encoder. Default as 'wavlm-base-plus'. Can switch to large backbones, such as WavLM-Large.
            num_ssl_feat: int
                Number of features in the uptream SSL representation. Default as 768, since we use the wavlm-base. Should be 1024 if using the large ones.
            num_fc_neurons: int
                Number of neurons in the downstream fully-connected (FC) layers. Default to be the same as 'num_ssl_feat'.
            num_classes: int
                Number of output neurons. Default as 1, since the diagnostic tasks are binary classification.
            freeze_encoder: Boolean
                Whether or not to freeze the upstream encoder. Default as True (only updating downstream modules).
            pooling_1: str
                Type of pooling to operate on the temporal representation. Available choices are ['avg','atn']. Default as 'atn', which is the attentive statistic pooling.
            pooling_2: str
                Type of pooling to operate on the modulation representation. Available choices are ['avg','atn']. Default as 'atn', which is the attentive statistic pooling.
            sample_rate: int
                Sampling rate of the upstream temporal representation. Default as 50, since the WavLM representation is sampled at 50Hz.
            win_length: int
                window length of the STFT (in ms) in the modulation block.
            hop_length: int
                hop length of the STFT (in ms) in the modulation block.
        """
        
        super().__init__(*args, **kwargs)

        # Upstream encoders
        self.ssl_encoder_source = ssl_encoder_source
        self.freeze_encoder = freeze_encoder
        self._init_upstream()

        # Modulation dynamics block
        self.dynamics = modulation_block(sr=sample_rate,win_len=win_length,hop_len=hop_length)

        # Pooling layers
        self.p1 = pooling_1
        self.p2 = pooling_2
        self.num_ssl_feat = num_ssl_feat
        self._init_pooling()

        # Classification head
        self.dp = dp
        self.num_classes = num_classes
        self.num_fc_neurons = num_fc_neurons
        self._init_clf_head()

        # Learnable layer weights
        self.weights_temporal = nn.Parameter(torch.ones(self.feature_extractor.config.num_hidden_layers))
        self.weights_dynamics = nn.Parameter(torch.ones(self.feature_extractor.config.num_hidden_layers))
        
    def _init_upstream(self):
        """
        Intializing the upstream encoder.
        """
        self.processor = AutoFeatureExtractor.from_pretrained(self.ssl_encoder_source)
        self.feature_extractor = WavLMModel.from_pretrained(self.ssl_encoder_source)
        for param in self.feature_extractor.parameters():
            param.requires_grad = not self.freeze_encoder

    def _init_clf_head(self):
        """
        Define classification head. FC+dropout+activation+FC.
        """
        num_temporal_feat = 2*self.num_ssl_feat if self.p1 == 'atn' else self.num_ssl_feat
        num_dynamics_feat = 2*self.num_ssl_feat if self.p2 == 'atn' else self.num_ssl_feat
        num_input_feat = num_temporal_feat + num_dynamics_feat
        if self.num_fc_neurons == -1: self.num_fc_neurons = num_input_feat
        self.fc = nn.Sequential(
            nn.Linear(num_input_feat, self.num_fc_neurons),
            nn.Dropout(p=self.dp),
            nn.LeakyReLU(0.1),
            nn.Linear(self.num_fc_neurons, self.num_classes)
        )

    def _init_pooling(self):
        """
        Define pooling functions.
        """
        if self.p1 == 'avg':
            self.pooling_layer_t = nn.AdaptiveAvgPool1d(1)
        elif self.p1 == 'atn':
            self.pooling_layer_t = AttentiveStatisticsPooling(self.num_ssl_feat,attention_channels=self.num_ssl_feat,global_context=True)
        if self.p2 == 'avg':
            self.pooling_layer_x = nn.AdaptiveAvgPool1d(1)
        elif self.p2 == 'atn':
            self.pooling_layer_x = AttentiveStatisticsPooling(self.num_ssl_feat,attention_channels=self.num_ssl_feat,global_context=True)

    def forward(self, x):
        # Upstream encoder processing
        input_values = self.processor(x, sampling_rate=16000, return_tensors="pt").input_values[0]
        input_values = input_values.to(device=x.device, dtype=x.dtype)
        features = self.feature_extractor(input_values, output_hidden_states=True)
        features = torch.stack(features.hidden_states[1:],dim=1) 
        B,L,T,F = features.shape # (batch, layer, time, features)

        # temporal branch:
        feat_t = self.weight_layer(features,branch='temporal',return_sum=True) # (batch, time, features)
        feat_t = feat_t.permute(0,2,1)
        feat_t = self.pooling_layer_t(feat_t).squeeze(-1)

        # dynamics branch:
        features = features.view(B*L,T,F)
        feat_x = self.dynamics(features)
        feat_x = feat_x.permute(0,2,1) # (B*L, F, freq)
        feat_x = feat_x.view(B,L,F,feat_x.shape[2])
        feat_x = self.weight_layer(feat_x,branch='dynamics',return_sum=True)
        feat_x = self.pooling_layer_x(feat_x).squeeze(-1)

        # Classification
        output = self.fc(torch.cat((feat_t,feat_x),axis=-1))
        output = output.view(output.shape[0],1)
        return output

    def weight_layer(self, features, branch, return_sum=False):
        """
        Assign learnable weights to different transformer layers. Can choose to return a weighted sum of all layers.
        Num_layers -> 1.
        """
        if torch.is_tensor(features):
            layer_num = features.shape[1]
            stacked_feature = features
        elif type(features) == tuple:
            layer_num = len(features)
            # Perform the weighted sum
            stacked_feature = torch.stack(features, dim=1)

        origin_shape = stacked_feature.shape
        stacked_feature = stacked_feature.reshape((layer_num, -1))
        if branch == 'temporal':
            norm_weights = F.softmax(self.weights_temporal, dim=-1)
        elif branch == 'dynamics':
            norm_weights = F.softmax(self.weights_dynamics, dim=-1)
        weighted_feature = (norm_weights.unsqueeze(-1) * stacked_feature)
        weighted_feature = weighted_feature.reshape(origin_shape)
        if return_sum:
            weighted_feature = weighted_feature.sum(dim=1)
        return weighted_feature

##############################################

class WeightPruner(object):
    """
    Can be used to prune fully-connected layers.
    
    Example:
    ------
    >>> pruner=WeightPruner(alpha=0.9)
    >>> model=WavRx()
    >>> model_modules['fc'][3].apply(pruner) # prune the last FC layer
    """
    def __init__(self,
                 alpha=0.9,
                 ):
        self.alpha=alpha
    
    def __call__(self,module):
        if hasattr(module,'weight'):
            w=module.weight.data
            threshold = self.alpha * torch.max(w)
            w_pruned = w * (torch.absolute(w) > threshold)
            module.weight.data=w_pruned