
import torch
import torch.nn as nn
import speechbrain as sb
from speechbrain.pretrained import EncoderClassifier
from speechbrain.processing.features import STFT
from speechbrain.processing.features import spectral_magnitude
from speechbrain.processing.features import Filterbank
from speechbrain.lobes.models.CRDNN import CRDNN
from speechbrain.nnet.pooling import Pooling1d


class feat_extract(nn.Module):

    def __init__(self,
                 fs:int,
                 win_length:int,
                 hop_length:int,
                 n_fft:int,
                 n_mels:int):
        
        super().__init__()
        self.compute_STFT = STFT(sample_rate=fs, win_length=win_length, hop_length=hop_length, n_fft=n_fft)
        self.compute_fbanks = Filterbank(n_mels=n_mels)

    def forward(self,x ):

        signal_STFT = self.compute_STFT(x)
        mag = spectral_magnitude(signal_STFT)
        fbanks = self.compute_fbanks(mag)
        return fbanks



class Classifier(sb.nnet.containers.Sequential):
    """This class implements the last MLP on the top of feature extracted from the encoder.
    Arguments
    ---------
    input_shape : tuple
        Expected shape of an example input.
    activation : torch class
        A class for constructing the activation layers.
    lin_blocks : int
        Number of linear layers.
    lin_neurons : int
        Number of neurons in linear layers.
    out_neurons : int
        Number of output neurons.

    Example
    -------
    >>> input_feats = torch.rand([5, 10, 40])
    >>> compute_xvect = Encoder()
    >>> xvects = compute_xvect(input_feats)
    >>> classify = Classifier(input_shape=xvects.shape)
    >>> output = classify(xvects)
    >>> output.shape
    torch.Size([5, 1, 1211])
    """

    def __init__(
        self,
        input_shape,
        activation=torch.nn.LeakyReLU,
        lin_blocks=1,
        lin_neurons=128,
        out_neurons=1,
    ):
        super().__init__(input_shape=input_shape)

        self.append(activation(), layer_name="act")
        self.append(sb.nnet.normalization.BatchNorm1d, layer_name="norm")

        if lin_blocks > 0:
            self.append(sb.nnet.containers.Sequential, layer_name="DNN")

        # Adding fully-connected layers
        for block_index in range(lin_blocks):
            block_name = f"block_{block_index}"
            self.DNN.append(
                sb.nnet.containers.Sequential, layer_name=block_name
            )
            self.DNN[block_name].append(
                sb.nnet.linear.Linear,
                n_neurons=lin_neurons,
                bias=True,
                layer_name="linear",
            )
            self.DNN[block_name].append(activation(), layer_name="act")
            self.DNN[block_name].append(
                sb.nnet.normalization.BatchNorm1d, layer_name="norm"
            )

        # Final Softmax classifier
        self.append(
            sb.nnet.linear.Linear, n_neurons=out_neurons, layer_name="out"
        )
        # self.append(
        #     sb.nnet.activations.Softmax(apply_log=True), layer_name="softmax"
        # )


class encoder(nn.Module):

    def __init__(self,
                 input_size:int,
                 cnn_channels:int,
                 rnn_layers:int,
                 rnn_neruons:int,
                 dnn_neurons:int):
        
        super().__init__()
        self.enc = CRDNN(input_size=input_size, 
                         cnn_channels=cnn_channels,
                         rnn_layers=rnn_layers,
                         rnn_neurons=rnn_neruons, 
                         dnn_neurons=dnn_neurons)

    def forward(self, x):
        output = self.enc(x)
        output = torch.mean(output, axis=1)
        return output

