"""
Speaker verification models, pretrained versions.

Author:
Yi Zhu
"""

import torch
import torch.nn as nn
import speechbrain as sb
from speechbrain.pretrained import EncoderClassifier


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
        lin_neurons=100,
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


class SV_clf(nn.Module):

    def __init__(self,
                clf_input_shape = (5, 1, 192),
                clf_activation = torch.nn.LeakyReLU,
                clf_lin_blocks = 1,
                clf_lin_neurons = 100,
                clf_out_neurons = 1,
                backbone_choice:str = 'ECAPA',
                pt_source:str = "speechbrain/spkrec-ecapa-voxceleb",
                mean_var_normalize_emb: bool = False,
                side_info:bool = False,
                freeze_extractor: bool = True,
                *args,
                **kwargs):
        
        super().__init__(*args,**kwargs)
        self.normalize = mean_var_normalize_emb
        self.side_info = side_info

        # Define backbone feature extractor
        assert backbone_choice in ['ECAPA', 'XVECTOR']
        if backbone_choice == 'ECAPA':
            self.feature_extractor = EncoderClassifier.from_hparams(source=pt_source) # remove ''run_opts={"device":"cuda"}'' to enable generating embeddings on cpu
        elif backbone_choice == 'XVECTOR':
            self.feature_extractor = EncoderClassifier.from_hparams(source=pt_source)
        
        # TODO: add side information
        # Define side information. Could be 'LFCC', 'MFCC', 'MSF', or any other speech features.
        if self.side_info:
            self.side_info_extractor = None
        
        # freeze parameters in encoders if necessary
        if freeze_extractor:
            for param in self.feature_extractor.parameters():
                param.require_grad = False
            if self.side_info:
                for param in self.side_info_extractor.parameters():
                    param.require_grad = False

        # Define classifier
        self.clf = Classifier(input_shape = clf_input_shape,
                              activation = clf_activation,
                              lin_blocks = clf_lin_blocks,
                              lin_neurons = clf_lin_neurons,
                              out_neurons = clf_out_neurons).to('cuda')
        
    def forward(self,x):

        # extract backbone features (and side features)
        backbone_output = self.feature_extractor.encode_batch(wavs=x, normalize=self.normalize)
        if self.side_info:
            side_output = self.side_info_extractor(x)
            clf_input = torch.cat([backbone_output, side_output], dim=-1)
        else: clf_input = backbone_output
        
        clf_input = clf_input.to(x.device) # copy to gpu
        # pass to the MLP classifier for final output
        output = self.clf(clf_input)

        return output
    
# if __name__ == '__main__':

#     clf = SV_clf(clf_input_shape=(5,1,192))
#     output = clf(torch.randn(5,8000))
#     print(output.shape)
