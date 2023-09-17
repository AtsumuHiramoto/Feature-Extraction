import torch
import torch.nn as nn

class BasicLSTM(nn.Module):
    """
    The most basic lstm structure.

    Parameters
    ----------
    in_dim (int): The number of expected features in the input x
    rec_dim (int): The number of features in the hidden state h
    out_dim (int): The number of outputs
    """
    def __init__(self,
                 in_dim,
                 rec_dim,
                 out_dim,
                 activation='tanh'):
        super(BasicLSTM, self).__init__()

        self.rnn = nn.LSTMCell(in_dim, rec_dim)
        self.rnn_out = nn.Sequential(
            nn.Linear(rec_dim, out_dim),
            activation
        )
    
    def forward(self, x, state=None):
        rnn_hid = self.rnn(x, state)
        y_hat   = self.rnn_out(rnn_hid[0])

        return y_hat, rnn_hid