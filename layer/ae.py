import torch
import torch.nn as nn

class BasicAE(nn.Module):
    """
    The most basic autoencoder structure.

    Parameters
    ----------
    in_dim (int): The number of expected features in the input x
    hid_dim (int): The number of features in the hidden state h
    """
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 activation='tanh'):
        super(BasicAE, self).__init__()
        
        if activation=="tanh":
            activation_function = nn.Tanh()
        elif activation=="sigmoid":
            activation_function = nn.Sigmoid()
        
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 100), activation_function,
            nn.Linear(100, hid_dim), activation_function
        )
        self.decoder = nn.Sequential(
            nn.Linear(hid_dim, 100), activation_function,
            nn.Linear(100, out_dim), activation_function
        )
    
    def forward(self, tac):
        # import ipdb; ipdb.set_trace()
        # import ipdb; ipdb.set_trace()
        hid = self.encoder(tac)
        y_tac  = self.decoder(hid)

        return y_tac, hid