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
        
        if activation=="tanh":
            activation_function = nn.Tanh()
        elif activation=="sigmoid":
            activation_function = nn.Sigmoid()

        self.rnn = nn.LSTMCell(in_dim, rec_dim)
        self.rnn_out = nn.Sequential(
            nn.Linear(rec_dim, out_dim),
            activation_function
        )
    
    def forward(self, tac, joint, state=None):
        # import ipdb; ipdb.set_trace()
        x = torch.cat([tac, joint], dim=1)
        # x = torch.cat([tac, joint]).reshape(1,-1)
        # import ipdb; ipdb.set_trace()
        rnn_hid = self.rnn(x, state)
        y_hat   = self.rnn_out(rnn_hid[0])
        yt_hat = y_hat[:,0:tac.shape[1]]
        yj_hat = y_hat[:,tac.shape[1]:]
        # import ipdb; ipdb.set_trace()

        return yt_hat, yj_hat, rnn_hid