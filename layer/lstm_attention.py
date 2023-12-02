import torch
import torch.nn as nn

class AttentionLSTM(nn.Module):
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
                 activation='tanh',
                 label=False):
        super(AttentionLSTM, self).__init__()
        
        if activation=="tanh":
            activation_function = nn.Tanh()
        elif activation=="sigmoid":
            activation_function = nn.Sigmoid()

        self.rnn = nn.LSTMCell(in_dim, rec_dim)
        self.rnn_out = nn.Sequential(
            nn.Linear(rec_dim, out_dim),
            activation_function
        )
        self.label = label
        self.rec_dim = rec_dim
        if label==True:
            self.rnn_out_label = nn.Sequential(
                nn.Linear(rec_dim, 10),
                nn.Sigmoid(),
                nn.Linear(10, 1),
                nn.Sigmoid()
            )
        self.attention_map = nn.Sequential(
                nn.Linear(rec_dim + in_dim, 100),
                nn.Sigmoid(),
                nn.Linear(100, in_dim),
                nn.Sigmoid()
        )
    
    def forward(self, tac, joint, torque=None, state=None):
        # import ipdb; ipdb.set_trace()
        if state is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            state = (torch.zeros([joint.shape[0], self.rec_dim]).to(device), torch.zeros([joint.shape[0], self.rec_dim]).to(device))
        if torque is None:
            x = torch.cat([tac, joint], dim=1)
        else:
            x = torch.cat([tac, joint, torque], dim=1)
        # x = torch.cat([tac, joint]).reshape(1,-1)
        mask_input = torch.cat([state[0], x], dim=1)
        # import ipdb; ipdb.set_trace()
        mask = self.attention_map(mask_input)
        x = torch.mul(mask, x)
        rnn_hid = self.rnn(x, state)
        y_hat   = self.rnn_out(rnn_hid[0])
        yt_hat = y_hat[:,0:tac.shape[1]]
        yj_hat = y_hat[:,tac.shape[1]:tac.shape[1]+joint.shape[1]]
        output = [yt_hat, yj_hat]
        if torque is not None:
            yp_hat = y_hat[:,tac.shape[1]+joint.shape[1]:tac.shape[1]+joint.shape[1]+torque.shape[1]]
            output.append(yp_hat)
        if self.label==True:
            yl_hat = self.rnn_out_label(rnn_hid[0])
            output.append(yl_hat)
        output.append(rnn_hid)
        output.append(mask)
        return output
        if torque is None:
            output.append(rnn_hid)
            return yt_hat, yj_hat, rnn_hid
        else:
            yp_hat = y_hat[:,tac.shape[1]+joint.shape[1]:tac.shape[1]+joint.shape[1]+torque.shape[1]]
            return yt_hat, yj_hat, yp_hat, rnn_hid
        # import ipdb; ipdb.set_trace()