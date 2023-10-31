import torch
import torch.nn as nn

class PatchFingerAE(nn.Module):
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
        super(PatchFingerAE, self).__init__()

        self.encoded_patch_num = 20
        self.encoded_finger_num = 20

        self.patch_range_list = [[i for i in range(30*3)], [i for i in range(30*3, 46*3)], [i for i in range(46*3, 62*3)], [i for i in range(62*3, 78*3)], 
                            [i for i in range(78*3, 108*3)], [i for i in range(108*3, 124*3)], [i for i in range(124*3, 140*3)], [i for i in range(140*3, 156*3)],
                            [i for i in range(156*3, 186*3)], [i for i in range(186*3, 202*3)], [i for i in range(202*3, 218*3)], [i for i in range(218*3, 234*3)],
                            [i for i in range(234*3, 264*3)], [i for i in range(264*3, 280*3)], [i for i in range(280*3, 296*3)],
                            [i for i in range(296*3, 320*3)], [i for i in range(320*3, 344*3)], [i for i in range(344*3, 368*3)]]
        self.patch_layout =  [[i for i in range(0, 4)],
                             [i for i in range(4, 8)],
                             [i for i in range(8, 12)],
                             [i for i in range(12, 15)],
                             [i for i in range(15, 18)]]
        self.finger_range_list = [[i for i in range(0, 78*3)],
                             [i for i in range(78*3, 156*3)],
                             [i for i in range(156*3, 234*3)],
                             [i for i in range(234*3, 296*3)],
                             [i for i in range(296*3, 368*3)]]
        
        if activation=="tanh":
            self.activation_function = nn.Tanh()
        elif activation=="sigmoid":
            self.activation_function = nn.Sigmoid()

        for i, patch_range in enumerate(self.patch_range_list):
            exec("self.patch_encoder_{} = nn.Sequential(nn.Linear(len(patch_range), self.encoded_patch_num), self.activation_function)".format(i))
            exec("self.patch_decoder_{} = nn.Sequential(nn.Linear(self.encoded_patch_num, len(patch_range)), self.activation_function)".format(i))
            # self.patch_decoder += nn.Sequential(nn.Linear(100, len(patch_range))).to("cuda")
        for i, patch_num in enumerate(self.patch_layout):
            exec("self.finger_encoder_{} = nn.Sequential(nn.Linear(self.encoded_patch_num*len(patch_num), self.encoded_finger_num), self.activation_function)".format(i))
            exec("self.finger_decoder_{} = nn.Sequential(nn.Linear(self.encoded_finger_num, self.encoded_patch_num*len(patch_num)), self.activation_function)".format(i))
            # self.finger_encoder += nn.Sequential(nn.Linear(100*len(patch_num), 100)).to("cuda")
            # self.finger_decoder += nn.Sequential(nn.Linear(100, 100*len(patch_num))).to("cuda")
        self.hand_encoder = nn.Sequential(nn.Linear(5*self.encoded_finger_num, hid_dim), self.activation_function)
        self.hand_decoder = nn.Sequential(nn.Linear(hid_dim, 5*self.encoded_finger_num), self.activation_function)
    
    def forward(self, tac):
        patch_tac = []
        for i, patch_range in enumerate(self.patch_range_list):
            tmp_tac = eval("self.patch_encoder_{}(tac[:, patch_range])".format(i))
            patch_tac.append(tmp_tac)
        finger_tac = []
        for i, patch_range in enumerate(self.patch_layout):
            tmp_tac = eval("self.finger_encoder_{}(torch.cat(patch_tac[patch_range[0] : patch_range[-1]+1], 1))".format(i))
            finger_tac.append(tmp_tac)
        hand_tac = torch.cat(finger_tac, 1)
        hid = self.hand_encoder(hand_tac)
        hand_tac = self.hand_decoder(hid)
        finger_tac = []
        for i in range(5):
            finger_tac.append(hand_tac[:,i*self.encoded_patch_num:(i+1)*self.encoded_patch_num])
        patch_tac = []
        for i, patch_range in enumerate(self.patch_layout):
            tmp_tac = eval("self.finger_decoder_{}(finger_tac[i])".format(i))
            for j in range(len(patch_range)):
                patch_tac.append(tmp_tac[:,j*self.encoded_finger_num:(j+1)*self.encoded_finger_num])
        
        y_tac = []
        for i, patch_range in enumerate(self.patch_range_list):
            tmp_tac = eval("self.patch_decoder_{}(patch_tac[i])".format(i))
            y_tac.append(tmp_tac)
        y_tac = torch.cat(y_tac, 1)

        return y_tac, hid