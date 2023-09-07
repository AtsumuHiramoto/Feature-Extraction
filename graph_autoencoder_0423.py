# coordinate of patch encoder : Fixed. not time series
# coordinate of hand encoder : time series. use real time coordinates
# add softmax into output of mlp
# add positional encoding into encoder
# without weight sharing
#reduce the num of encoder's layer 04/12
# concat instead of convolution
import math
import torch
import torch.nn as nn
import torch_geometric
# torch.set_default_tensor_type(torch.cuda.FloatTensor)

class PatchEncoder(nn.Module):
    def __init__(self, channel=[3,8,16]):
        super().__init__()
        self.channel = channel
        # self.continuous_kernel1 = nn.Sequential(
        #     nn.Linear(2,32), nn.InstanceNorm1d(32), nn.ReLU(),
        #     nn.Linear(32,64), nn.InstanceNorm1d(64), nn.ReLU(),
        #     nn.Linear(64,self.channel[0]*self.channel[1])
        # )
        # self.continuous_kernel1 = nn.Sequential(
        #     nn.Linear(2,8), nn.ReLU(),
        #     nn.Linear(8,self.channel[0]*self.channel[1])
        # )
        self.continuous_kernel = nn.Sequential(
            nn.Linear(2,8), nn.ReLU(),
            nn.Linear(8,16), nn.ReLU(),
            # nn.Linear(16,32), nn.ReLU(),
            nn.Linear(16,self.channel[1])
        )

        self.encoder = nn.Linear(self.channel[0]+self.channel[1], self.channel[2])

        # self.continuous_kernel1 = nn.Sequential(
        #     nn.Linear(2,8), nn.ReLU(),
        #     nn.Linear(8,16), nn.ReLU(),
        #     nn.Linear(16,32), nn.ReLU(),
        #     nn.Linear(32,self.channel[0]*self.channel[1])
        # )
        # self.continuous_kernel2 = nn.Sequential(
        #     nn.Linear(2,32), nn.InstanceNorm1d(32), nn.ReLU(),
        #     nn.Linear(32,64), nn.InstanceNorm1d(64), nn.ReLU(),
        #     nn.Linear(64,self.channel[1]*self.channel[2])
        # )
        # self.continuous_kernel2 = nn.Sequential(
        #     nn.Linear(2,8), nn.ReLU(),
        #     nn.Linear(8,self.channel[1]*self.channel[2])
        # )
        self.relu = nn.ReLU()
    # coordinates : [10,16,3] -> [160,3], [10,16,3,8]
    def forward(self, feature, coordinates):
        x_list = []
        patch_node = []
        for node_index in range(len(coordinates)):
            # calc relative coordinate
            r_coordinates = coordinates - coordinates[node_index]
            # print(r_coordinates)
            # import ipdb; ipdb.set_trace()
            kernel = self.continuous_kernel(r_coordinates) #16 [16,128]
            # import ipdb; ipdb.set_trace()
            feature_list = []
            for b in range(feature.shape[0]):
                tmp_feature = torch.cat([feature[b,node_index,:], kernel[node_index,:]])
                feature_list.append(tmp_feature)
            x = torch.stack(feature_list)
            x = self.encoder(x)
            x_list.append(x)
        # import ipdb; ipdb.set_trace()        
        # for i, continuous_kernel in enumerate([self.continuous_kernel1, self.continuous_kernel2]):
        #     x_list = []
        #     patch_node = []
        #     for node_index in range(len(coordinates)):
        #         # calc relative coordinate
        #         r_coordinates = coordinates - coordinates[node_index]
        #         # print(r_coordinates)
        #         # import ipdb; ipdb.set_trace()
        #         kernel = continuous_kernel(r_coordinates) #16 [16,128]
        #         # import ipdb; ipdb.set_trace()
        #         feature_list = []
        #         for b in range(feature.shape[0]):
        #             tmp_feature = torch.cat([feature[b,node_index,:], kernel[node_index,:]])
        #             feature_list.append(tmp_feature)
        #         x = torch.stack(feature_list)
        #         x_list.append(x)
        #     import ipdb; ipdb.set_trace()
        '''

                # convolution process
                # import ipdb; ipdb.set_trace()
                x = torch.bmm(feature.permute(1,0,2), kernel.view(-1,self.channel[i],self.channel[i+1])) # [16,10,3]x[16,3,8] = [16,10,8] 
                x = x.sum(dim=0) # [10,8]
                #activation
                x = self.relu(x)

                patch_node.append(x)
            feature = torch.stack(patch_node, dim=1) # [10,16,8]
            # import ipdb; ipdb.set_trace()
        # pooling process
        feature, _ = feature.max(dim=1) # [10,8] => [10,16]
        # import ipdb; ipdb.set_trace()

        return feature
            # # [10,16,3]x[10,3,8] = [10,16,8] 

            # x = feature[:,:,0] * kernel.flatten() # adamal product : [10,16],[16,24] => [10,16,24]
            # x = x.sum(dim=1)
            # # activation
            # x = self.relu(x)
            # # import ipdb; ipdb.set_trace()
            # patch_node.append(x)
        # import ipdb; ipdb.set_trace()
        # patch_node = torch.stack(patch_node, dim=1) # N*16node
        # # pooling
        # patch_node, _ = patch_node.max(dim=1) # N*1
        # return patch_node
        '''

class HandEncoder(nn.Module):
    def __init__(self, channel=[16,32,64]):
        super().__init__()
        # self.input_channel = input_channel
        # self.output_channel = output_channel
        # self.continuous_kernel = nn.Sequential(
        #     nn.Linear(3,32), nn.InstanceNorm1d(32), nn.ReLU(),
        #     nn.Linear(32,16), nn.InstanceNorm1d(16), nn.ReLU(),
        #     nn.Linear(16,self.input_channel*self.output_channel)
        # )
        self.channel = channel
        # self.continuous_kernel1 = nn.Sequential(
        #     nn.Linear(3,32), nn.InstanceNorm1d(32), nn.ReLU(),
        #     nn.Linear(32,64), nn.InstanceNorm1d(64), nn.ReLU(),
        #     nn.Linear(64,self.channel[0]*self.channel[1])
        # )
        # self.continuous_kernel1 = nn.Sequential(
        #     nn.Linear(3,8), nn.ReLU(),
        #     nn.Linear(8,self.channel[0]*self.channel[1])
        # )
        self.continuous_kernel1 = nn.Sequential(
            nn.Linear(3,8), nn.ReLU(),
            nn.Linear(8,16), nn.ReLU(),
            nn.Linear(16,32), nn.ReLU(),
            nn.Linear(32,self.channel[0]*self.channel[1])
        )
        # self.continuous_kernel2 = nn.Sequential(
        #     nn.Linear(3,32), nn.InstanceNorm1d(32), nn.ReLU(),
        #     nn.Linear(32,64), nn.InstanceNorm1d(64), nn.ReLU(),
        #     nn.Linear(64,self.channel[1]*self.channel[2])
        # )
        # self.continuous_kernel2 = nn.Sequential(
        #     nn.Linear(3,8), nn.ReLU(),
        #     nn.Linear(8,self.channel[1]*self.channel[2])
        # )
        self.continuous_kernel2 = nn.Sequential(
            nn.Linear(3,8), nn.ReLU(),
            nn.Linear(8,16), nn.ReLU(),
            nn.Linear(16,32), nn.ReLU(),
            nn.Linear(32,self.channel[1]*self.channel[2])
        )
        self.relu = nn.ReLU()
        # feature : [10,22,16]
    # coordinates : [10,22,3] -> [220,3] --mlp--> [220,16x32] -> [10x22,16,32]
    def forward(self, feature, coordinates):
        batch_size, patch_num, channel = feature.shape
        for i, continuous_kernel in enumerate([self.continuous_kernel1, self.continuous_kernel2]):
            patch_node = []
            for node_index in range(patch_num):
                # calc relative coordinate
                # import ipdb; ipdb.set_trace()
                r_coordinates = coordinates - coordinates[:,node_index,:].view(batch_size,1,-1) #[10,22,3]
                # print(r_coordinates)
                # import ipdb; ipdb.set_trace()
                r_coordinates = r_coordinates.view(-1,3) # [220,3]
                kernel = continuous_kernel(r_coordinates) # [220,512]
                # import ipdb; ipdb.set_trace()

                # convolution process
                x = torch.bmm(feature.view(-1,1,self.channel[i]), kernel.view(-1,self.channel[i],self.channel[i+1])) # [220,1,16]x[220,16,32] = [220,1,32]
                x = x.view(batch_size,patch_num,self.channel[i+1]) # [10,22,32]
                x = x.sum(dim=1) # [10,32]
                #activation
                x = self.relu(x)

                patch_node.append(x)
            # import ipdb; ipdb.set_trace()
            feature = torch.stack(patch_node, dim=1) # [10,22,32]
            # import ipdb; ipdb.set_trace()
        # pooling process
        feature, _ = feature.max(dim=1) # [10,22,64] => [10,64]

        return feature

# input x=10*1*16 output [y_1,y_2,...,y_20]=10*20*8
# y_n=mlp([x,v_n])
# v_n : vector from hand_CoG to nth patch_CoG
class HandDecoder(nn.Module):
    def __init__(self, channel=[64,32,16]):
        super().__init__()
        self.channel = channel
        # decode MLP :y = g(f,v)
        # f : encoded feature, v : vector of relative coordinates 
        # self.decode_mlp = nn.Sequential(
        #     nn.Linear(self.channel[0]+3,32), nn.InstanceNorm1d(32), nn.ReLU(),
        #     nn.Linear(32,16), nn.InstanceNorm1d(16), nn.ReLU(),
        #     nn.Linear(16,self.channel[1])
        # )
        self.decode_mlp = nn.Sequential(
            nn.Linear(self.channel[0]+3,32), nn.ReLU(),
            nn.Linear(32,16), nn.ReLU(),
            nn.Linear(16,self.channel[1])
        )
        self.continuous_kernel1 = nn.Sequential(
            nn.Linear(3,8), nn.ReLU(),
            nn.Linear(8,16), nn.ReLU(),
            nn.Linear(16,self.channel[1]*self.channel[2])
        )
        self.relu = nn.ReLU()
    # feature : N*channel [10,64]
    # coordinates_v : vector from hand_CoG to each patch_CoG 20*3 [10,22,3]
    def forward(self, feature, coordinates_v):
        reconstructed_patch = []
        coordinates_v = coordinates_v.permute(1,0,2) # [22,10,3]
        for v in coordinates_v:
            # vector = torch.stack([v for i in range(feature.shape[0])], dim=0) # N*3
            # x = torch.stack([feature, vector], dim=1) # N*(10+3)
            x = torch.cat([feature, v], dim=1) # [10,64+3]
            x = self.decode_mlp(x) # N*8
            x = self.relu(x)
            reconstructed_patch.append(x)
        reconstructed_patch = torch.stack(reconstructed_patch, dim=1) # N*20*8
        # import ipdb; ipdb.set_trace()

        feature = reconstructed_patch
        coordinates_v = coordinates_v.permute(1,0,2)
        for i, continuous_kernel in enumerate([self.continuous_kernel1]):
            patch_node = []
            batch_size, patch_num, channel = feature.shape

            for node_index in range(patch_num):
                # calc relative coordinate
                # import ipdb; ipdb.set_trace()
                r_coordinates = coordinates_v - coordinates_v[:,node_index,:].view(batch_size,1,-1) #[10,22,3]
                # print(r_coordinates)
                # import ipdb; ipdb.set_trace()
                r_coordinates = r_coordinates.view(-1,3) # [220,3]
                kernel = continuous_kernel(r_coordinates) # [220,512]
                # import ipdb; ipdb.set_trace()

                # convolution process
                x = torch.bmm(feature.view(-1,1,self.channel[i+1]), kernel.view(-1,self.channel[i+1],self.channel[i+2])) # [220,1,16]x[220,16,32] = [220,1,32]
                x = x.view(batch_size,patch_num,self.channel[i+2]) # [10,22,32]
                x = x.sum(dim=1) # [10,32]
                #activation
                x = self.relu(x)

                patch_node.append(x)
            feature = torch.stack(patch_node, dim=1) # [10,22,32]
            # import ipdb; ipdb.set_trace()
        # import ipdb; ipdb.set_trace()
        return feature
        # return reconstructed_patch # [10,22,32]

# input x=10*1*16 output [y_1,y_2,...,y_20]=10*20*8 #INPUT MUST BE [-1,1] TO USE NERF2D
# y_n=mlp([x,v_n])
# v_n : vector from hand_CoG to nth patch_CoG
class HandDecoder_nerf2d(nn.Module):
    def __init__(self, channel=[64,32,16], L=10):
        super().__init__()
        self.channel = channel
        # decode MLP :y = g(f,v)
        # f : encoded feature, v : vector of relative coordinates 
        # self.decode_mlp = nn.Sequential(
        #     nn.Linear(self.channel[0]+3,32), nn.InstanceNorm1d(32), nn.ReLU(),
        #     nn.Linear(32,16), nn.InstanceNorm1d(16), nn.ReLU(),
        #     nn.Linear(16,self.channel[1])
        # )
        self.decode_mlp = nn.Sequential(
            nn.Linear(self.channel[0]+3*2*L,32), nn.ReLU(),
            nn.Linear(32,16), nn.ReLU(),
            nn.Linear(16,self.channel[1])
        )
        self.continuous_kernel1 = nn.Sequential(
            nn.Linear(3,8), nn.ReLU(),
            nn.Linear(8,16), nn.ReLU(),
            nn.Linear(16,self.channel[1]*self.channel[2])
        )
        self.relu = nn.ReLU()
    # feature : N*channel [10,64]
    # coordinates_v : vector from hand_CoG to each patch_CoG 20*3 [10,22,3]
    def forward(self, feature, coordinates_v_pe, coordinates_v):
        reconstructed_patch = []
        coordinates_v_pe = coordinates_v_pe.permute(1,0,2) # [22,10,63*2*L]

        for v in coordinates_v_pe:
            # vector = torch.stack([v for i in range(feature.shape[0])], dim=0) # N*3
            # x = torch.stack([feature, vector], dim=1) # N*(10+3)
            x = torch.cat([feature, v], dim=1) # [10,64+3*2*L]
            x = self.decode_mlp(x) # N*8
            x = self.relu(x)
            reconstructed_patch.append(x)
        reconstructed_patch = torch.stack(reconstructed_patch, dim=1) # N*20*8
        # import ipdb; ipdb.set_trace()

        feature = reconstructed_patch
        # coordinates_v = coordinates_v.permute(1,0,2)
        for i, continuous_kernel in enumerate([self.continuous_kernel1]):
            patch_node = []
            batch_size, patch_num, channel = feature.shape

            for node_index in range(patch_num):
                # calc relative coordinate
                # import ipdb; ipdb.set_trace()
                r_coordinates = coordinates_v - coordinates_v[:,node_index,:].view(batch_size,1,-1) #[10,22,3]
                # print(r_coordinates)
                # import ipdb; ipdb.set_trace()
                r_coordinates = r_coordinates.view(-1,3) # [220,3]
                kernel = continuous_kernel(r_coordinates) # [220,512]
                # import ipdb; ipdb.set_trace()

                # convolution process
                x = torch.bmm(feature.view(-1,1,self.channel[i+1]), kernel.view(-1,self.channel[i+1],self.channel[i+2])) # [220,1,16]x[220,16,32] = [220,1,32]
                x = x.view(batch_size,patch_num,self.channel[i+2]) # [10,22,32]
                x = x.sum(dim=1) # [10,32]
                #activation
                x = self.relu(x)

                patch_node.append(x)
            feature = torch.stack(patch_node, dim=1) # [10,22,32]
            # import ipdb; ipdb.set_trace()
        # import ipdb; ipdb.set_trace()
        return feature
        # return reconstructed_patch # [10,22,32]

class PatchDecoder(nn.Module):
    def __init__(self, channel=[32,3]):
        super().__init__()
        self.channel = channel
        # decode MLP :y = g(f,v)
        # f : encoded feature, v : vector of relative coordinates 
        # self.decode_mlp = nn.Sequential(
        #     nn.Linear(self.channel[0]+2,32), nn.InstanceNorm1d(32), nn.ReLU(),
        #     nn.Linear(32,16), nn.InstanceNorm1d(16), nn.ReLU(),
        #     nn.Linear(16,self.channel[1])
        # )
        self.decode_mlp = nn.Sequential(
            nn.Linear(self.channel[0]+2,32), nn.ReLU(),
            nn.Linear(32,16), nn.ReLU(),
            nn.Linear(16,self.channel[1])
        )
        self.relu = nn.ReLU()
    # feature : N*channel [10,32]
    # coordinates_v : vector from hand_CoG to each patch_CoG 20*3 [16,2]
    def forward(self, feature, coordinates_v):
        reconstructed_patch = []
        # coordinates_v = coordinates_v.permute(1,0,2)
        for v in coordinates_v:
            vector = torch.stack([v for i in range(feature.shape[0])], dim=0) # N*2
            # x = torch.stack([feature, vector], dim=1) # N*(10+2)
            # import ipdb;ipdb.set_trace()
            x = torch.cat([feature, vector], dim=1) # [10,32+2]
            x = self.decode_mlp(x) # N*8
            # x = self.relu(x)
            import ipdb;ipdb.set_trace()
            reconstructed_patch.append(x)
        reconstructed_patch = torch.stack(reconstructed_patch, dim=1) # N*16*3
        # import ipdb; ipdb.set_trace()
        return reconstructed_patch # [10,16,3]

class PatchDecoder_nerf2d(nn.Module):
    def __init__(self, channel=[32,3], L=10):
        super().__init__()
        self.channel = channel
        # decode MLP :y = g(f,v)
        # f : encoded feature, v : vector of relative coordinates 
        # self.decode_mlp = nn.Sequential(
        #     nn.Linear(self.channel[0]+2,32), nn.InstanceNorm1d(32), nn.ReLU(),
        #     nn.Linear(32,16), nn.InstanceNorm1d(16), nn.ReLU(),
        #     nn.Linear(16,self.channel[1])
        # )
        self.L = L
        self.decode_mlp = nn.Sequential(
            nn.Linear(self.channel[0]+2*L,32), nn.ReLU(),
            nn.Linear(32,16), nn.ReLU(),
            nn.Linear(16,self.channel[1])
        )
        self.relu = nn.ReLU()
    # feature : N*channel [10,32]
    # coordinates_v : vector from hand_CoG to each patch_CoG 20*3 [16,2]
    def forward(self, feature, coordinates_v):
        reconstructed_patch = []
        # coordinates_v = coordinates_v.permute(1,0,2)
        for v in coordinates_v:
            v = positional_encoding(v, L=self.L)
            vector = torch.stack([v for i in range(feature.shape[0])], dim=0) # N*2
            # x = torch.stack([feature, vector], dim=1) # N*(10+2)
            # import ipdb;ipdb.set_trace()
            x = torch.cat([feature, vector], dim=1) # [10,32+2]
            x = self.decode_mlp(x) # N*8
            # x = self.relu(x)
            # import ipdb;ipdb.set_trace()
            reconstructed_patch.append(x)
        reconstructed_patch = torch.stack(reconstructed_patch, dim=1) # N*16*3
        # import ipdb; ipdb.set_trace()
        return reconstructed_patch # [10,16,3]
        
class ContinuousCAE(nn.Module):
    # coordinates_node: patch*node*channel (e.g. 22*[16 or 30]*2)
    def __init__(self, channel_patch=[3,8,16], channel_hand=[16,32,64], decode_pe_flag=1, cfg=None):
        super().__init__()
        self.channel_patch = cfg["model"]["channel_patch"]
        self.channel_hand = cfg["model"]["channel_hand"]
        self.channel_patch_decode = cfg["model"]["channel_patch_decode"]
        self.num_patch = cfg["data"]["num_patch"]
        self.num_data = cfg["data"]["num_data"]
        self.decode_pe_flag = cfg["model"]["PositionalEncoding"]["decode_pe_flag"]
        self.pe_patch_decoder_flag = cfg["model"]["PositionalEncoding"]["patch_decoder"]
        self.L = cfg["model"]["PositionalEncoding"]["L"]
        self.weight_sharing = cfg["model"]["weight_sharing"]
        self.fnn_without_weightshare = cfg["model"]["fnn_without_weightshare"]
        self.decoder_without_weightshare = cfg["model"]["decoder_without_weightshare"]
        self.debug = cfg["model"]["debug"]
        # self.coordinates_patch = coordinates_patch
        self.coordinates_patch = make_patch_coordinates()
        if self.fnn_without_weightshare!=None:
            for i in range(int(self.num_data/3)):
                for j in range(len(self.fnn_without_weightshare)-1):
                    exec("self.fnn_{}_{} = nn.Linear({},{})".format(i, j, self.fnn_without_weightshare[j], self.fnn_without_weightshare[j+1]))
        if self.debug["patchencoder"]==1:
            if self.weight_sharing:
                self.patchencoder = PatchEncoder(channel=self.channel_patch)
            else:
                for i in range(self.num_patch):
                    exec("self.patchencoder_{} = PatchEncoder(channel=self.channel_patch)".format(i))
        else:
            self.patchencoder_fnn = nn.Sequential(
                nn.Linear(self.num_data, self.num_patch*self.channel_patch[-1]), nn.ReLU()
            )
        if self.debug["handencoder"]==1:
            self.handencoder = HandEncoder(channel=self.channel_hand)
        else:
            self.handencoder_fnn = nn.Sequential(
                nn.Linear(self.num_patch*self.channel_patch[-1], self.channel_hand[-1]), nn.ReLU()
            )
        if self.debug["handdecoder"]==1:
            if self.decode_pe_flag:
                self.handdecoder = HandDecoder_nerf2d(channel=self.channel_hand[::-1])
            else:
                self.handdecoder = HandDecoder(channel=self.channel_hand[::-1])
        else:
            # self.handdecoder_fnn = nn.Sequential(
            #     nn.Linear(self.channel_hand[-1], self.num_patch*self.channel_patch[-1]), nn.ReLU()
            # )
            self.handdecoder_fnn = nn.Sequential(
                nn.Linear(self.channel_hand[-1], self.channel_hand[-1]*2), nn.ReLU(), 
                nn.Linear(self.channel_hand[-1]*2, self.num_patch*self.channel_patch[-1]), nn.ReLU()
            )
        if self.debug["patchdecoder"]==1:
            # self.patchdecoder = PatchDecoder(channel=self.channel_patch[::-1])
            if self.decoder_without_weightshare:
                for i in range(self.num_patch):
                    if self.pe_patch_decoder_flag:
                        exec("self.patchdecoder_{} = PatchDecoder_nerf2d(channel=self.channel_patch_decode, L=self.L)".format(i))
                    else:
                        exec("self.patchdecoder_{} = PatchDecoder(channel=self.channel_patch_decode)".format(i))
            else:
                if self.pe_patch_decoder_flag:
                    self.patchdecoder = PatchDecoder_nerf2d(channel=self.channel_patch_decode, L=self.L)
                else:
                    self.patchdecoder = PatchDecoder(channel=self.channel_patch_decode)
        else:
            # self.patchdecoder_fnn = nn.Sequential(
            #     nn.Linear(self.num_patch*self.channel_patch[-1], self.num_data), nn.ReLU()
            # )
            self.patchdecoder_fnn = nn.Sequential(
                nn.Linear(self.num_patch*self.channel_patch_decode[0], self.num_patch*self.channel_patch_decode[0]*2), nn.ReLU(),
                nn.Linear(self.num_patch*self.channel_patch_decode[0]*2, self.num_data),
            )
        # self.handdecoder = HandDecoder(channel=[64,32,16]) # channel_hand[::-1]
        # self.patchdecoder = PatchDecoder(channel=[32,3])
        # self.patchdecoder = PatchDecoder(channel=[16,3])
        # self.fnn = nn.Linear(self.channel_patch[-1]*22,1152)
        # self.fnn1 = nn.Linear(self.channel_hand[-1],1152)
        # self.decoder = nn.Sequential(
        #     nn.Linear(self.channel_hand[-1], 128), nn.ReLU(),
        #     nn.Linear(128, 256), nn.ReLU(),
        #     nn.Linear(256,1152)
        # )
        # self.decoder_ = nn.Sequential(
        #     nn.Linear(22*self.channel_hand[-3], 128), nn.ReLU(),
        #     nn.Linear(128, 256), nn.ReLU(),
        #     nn.Linear(256,1152)
        # )
    
    # feature: [100,1152]
    # feature_list: patch*batch*node*channel (e.g. 22*1000*[16 or 30]*3)
    # coordinates_cog: batch*patch*channel (e.g. 10*22*3) <- center of gravity for each patches
    def forward(self, feature, coordinates_cog, coordinates_cog_pe=None):
        # [100,1152]
        # if self.fnn_without_weightshare!=None:
        #     tmp_feature_list = []
        #     for i in range(int(self.num_data/3)):
        #         tmp_feature = feature[:,i*3:(i+1)*3]
        #         for j in range(len(self.fnn_without_weightshare)-1):
        #             tmp_feature = eval("self.fnn_{}_{}(tmp_feature])".format(i, j))
        #         tmp_feature_list.append(tmp_feature)
        #     feature = torch.cat(tmp_feature_list, dim = )
        self.timestep = feature.shape[0]
        if self.debug["patchencoder"]==1:
            feature_list = make_input_data(feature)
            # import ipdb; ipdb.set_trace()
            if self.fnn_without_weightshare!=None:
                tmp_feature_list = []
                count = 0
                for i in range(self.num_patch): #22
                    tmp_feature_list_patch = []
                    for j in range(feature_list[i].shape[1]): #16~
                        tmp_feature = feature_list[i][:,j,:]
                        for k in range(len(self.fnn_without_weightshare)-1):
                            tmp_feature = eval("self.fnn_{}_{}(tmp_feature)".format(count, k))
                        tmp_feature_list_patch.append(tmp_feature)
                        count += 1
                    tmp_feature_list_patch = torch.stack(tmp_feature_list_patch, dim=1)
                    tmp_feature_list.append(tmp_feature_list_patch)
                # import ipdb; ipdb.set_trace()
                feature_list = tmp_feature_list
            # import ipdb; ipdb.set_trace() # 22x[100,16,3] => 22x[100,16,12]
            num_patch = len(feature_list)
            patch_feature_list = []
            if self.weight_sharing:
                for patch_index in range(num_patch):
                    patch_feature = self.patchencoder(feature_list[patch_index], self.coordinates_patch[patch_index])
                    patch_feature_list.append(patch_feature)
            else:
                for patch_index in range(num_patch):
                    patch_feature = eval("self.patchencoder_{}(feature_list[patch_index], self.coordinates_patch[patch_index])".format(patch_index))
                    patch_feature_list.append(patch_feature)
            import ipdb; ipdb.set_trace()          
            patch_feature_list = torch.stack(patch_feature_list, dim=1) # N*patch*channel
        else:
            feature = self.patchencoder_fnn(feature)
            # patch_feature_list = [feature[:,self.channel_patch[-1]*i:self.channel_patch[-1]*(i+1)] for i in range(self.num_patch)]
            patch_feature_list = feature.view(-1, self.num_patch, self.channel_patch[-1])
        # import ipdb; ipdb.set_trace() # [100,22,16]
        # x = patch_feature_list.view(100,-1)
        # patch_feature_list = self.fnn(x)
        if self.debug["handencoder"]==1:
            hand_feature = self.handencoder(patch_feature_list, coordinates_cog)
        else:
            # import ipdb; ipdb.set_trace()
            patch_feature_list = patch_feature_list.view(-1,self.num_patch*self.channel_patch[-1])
            hand_feature = self.handencoder_fnn(patch_feature_list)
        # N*channel [10,64]
        # import ipdb; ipdb.set_trace() # [100,64]
        # patch_feature_list = self.fnn1(hand_feature)
        # patch_feature_list = self.decoder(hand_feature)
        if self.debug["handdecoder"]==1:
            if self.decode_pe_flag:
                x = self.handdecoder(hand_feature, coordinates_cog_pe, coordinates_cog) # [10,22,32]
            else:
                x = self.handdecoder(hand_feature, coordinates_cog)
        else:
            x = self.handdecoder_fnn(hand_feature)
        # import ipdb; ipdb.set_trace()
        # x = x.view(100,-1)
        # patch_feature_list = self.decoder_(x)        
        # # hand_feature=0
        # # x = patch_feature_list
        # # import ipdb; ipdb.set_trace() # [100,22,32]
        if self.debug["patchdecoder"]==1:
            patch_feature_list = []
            for patch in range(self.num_patch):
                if self.decoder_without_weightshare:
                    patch_feature = eval("self.patchdecoder_{}(x[:,patch,:], self.coordinates_patch[patch])".format(patch))
                # import ipdb; ipdb.set_trace()
                else:
                    patch_feature = self.patchdecoder(x[:,patch,:], self.coordinates_patch[patch]) # [10,16,3]
                patch_feature_list.append(patch_feature)
            # import ipdb; ipdb.set_trace()
            patch_feature_list = make_output_data(patch_feature_list) # [10,22,16,3]
        else:
            # import ipdb;ipdb.set_trace() 
            x = x.view(self.timestep, -1)
            patch_feature_list  = self.patchdecoder_fnn(x)
        # import ipdb;ipdb.set_trace() # 22x[100,16,3]
      
        # hand_feature=0
        # patch_feature_list = make_output_data(feature_list)
        # import ipdb; ipdb.set_trace()
        return hand_feature, patch_feature_list

def positional_encoding(v, L=10):
    encoded_v = []
    for l in range(L):
        val = 2**l
        # import ipdb; ipdb.set_trace()
        encoded_v.append(torch.sin(val*v[0]*math.pi))
        encoded_v.append(torch.cos(val*v[1]*math.pi))
    # import ipdb; ipdb.set_trace()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoded_v = torch.tensor(encoded_v).to(device)
    return encoded_v

def make_output_data(patch_feature_list):
    output_list = []
    batch_num = patch_feature_list[0].shape[0]
    for i in range(len(patch_feature_list)):
        # if i==0:
        #     output = patch_feature_list[i].view(batch_num,-1)
        # else:
        #     output = torch.cat(output, patch_feature_list[i].view(batch_num,-1))
        output = patch_feature_list[i].view(batch_num,-1)
        output_list.append(output)
    # import ipdb;ipdb.set_trace()
    output_list = torch.cat(output_list, 1)

    return output_list

def make_patch_coordinates():
    patch_all = []
    patch_1 = torch.tensor(
        [[3,0], [3,1], [3,2], [3,3],
        [2,0], [2,1], [2,2], [2,3],
        [1,0], [1,1], [1,2], [1,3],
        [0,0], [0,1], [0,2], [0,3],
        ]
    , dtype=torch.float32); patch_all.append(patch_1)
    patch_2 = torch.tensor(
        [[3,0], [3,2], [3,1], [3,3],
        [2,0], [2,2], [2,1], [2,3],
        [1,0], [1,2], [1,1], [1,3],
        [0,0], [0,2], [0,1], [0,3],
        ]
    , dtype=torch.float32); patch_all.append(patch_2)
    patch_3 = patch_2.clone(); patch_all.append(patch_3)
    patch_4_5 = torch.tensor(
        [[2,3], [2,1], [2,2], [2,0], 
        [2,4], [1,4], [2,5], [3,4], 
        [0,3], [0,1], [0,2], [0,0],
        [1,3], [1,1], [1,2], [1,0],
        [3,3], [3,1], [3,2], [3,0],
        [4,3], [4,1], [4,2], [4,0],
        ]
    , dtype=torch.float32); patch_all.append(patch_4_5)
    patch_6 = patch_1.clone(); patch_all.append(patch_6)
    patch_7 = torch.tensor(
        [[0,3], [0,2], [0,1], [0,0],
        [1,3], [1,2], [1,1], [1,0],
        [2,3], [2,2], [2,1], [2,0],
        [3,3], [3,2], [3,1], [3,0],
        ]
    , dtype=torch.float32); patch_all.append(patch_7)
    patch_8 = patch_7.clone(); patch_all.append(patch_8)
    patch_9_10 = patch_4_5.clone(); patch_all.append(patch_9_10)
    patch_11 = patch_1.clone(); patch_all.append(patch_11)
    patch_12 = patch_7.clone(); patch_all.append(patch_12)
    patch_13 = patch_7.clone(); patch_all.append(patch_13)
    patch_14_15 = patch_4_5.clone(); patch_all.append(patch_14_15)
    patch_16 = patch_1.clone(); patch_all.append(patch_16)
    patch_17 = patch_1.clone(); patch_all.append(patch_17)
    patch_18_19 = patch_4_5.clone(); patch_all.append(patch_18_19)
    patch_20 = patch_7.clone(); patch_all.append(patch_20)
    patch_21 = patch_7.clone(); patch_all.append(patch_21)
    patch_22 = patch_7.clone(); patch_all.append(patch_22)
    patch_23 = patch_7.clone(); patch_all.append(patch_23)
    patch_24 = patch_7.clone(); patch_all.append(patch_24)
    patch_25 = patch_1.clone(); patch_all.append(patch_25)
    patch_26 = patch_1.clone(); patch_all.append(patch_26)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for i in range(len(patch_all)):
        patch_all[i] = (patch_all[i]/4).to(device)
    return patch_all

def main():
    patch_all = make_patch_coordinates()
    # # import ipdb; ipdb.set_trace()
    # patch_phalange_f = torch.rand(10,4,4,3) # N*H*W*C
    # patch_phalange_f = patch_phalange_f.reshape(10,-1,3)
    # # print(patch_phalange_f)
    # # patch_phalange_c = torch.tensor(
    # #     [[[-3,3],[-1,3],[1,3],[3,3]],
    # #     [[-3,1],[-1,1],[1,1],[3,1]],
    # #     [[-3,-1],[-1,-1],[1,-1],[3,-1]],
    # #     [[-3,-3],[-1,-3],[1,-3],[3,-3]]], dtype=torch.float32) # [4,4,2]
    # patch_phalange_c = torch.rand([4,4,3])
    # print(patch_phalange_c.shape)
    # patch_phalange_c = patch_phalange_c.reshape(-1,2) #[16,2]
    # # print(patch_phalange_c.shape)
    # output_channel = [8,16]
    # model = PatchEncoder(output_channel=output_channel[0])
    # y = model(patch_phalange_f, patch_phalange_c)

    # patch_hand_c = torch.rand([20,3]) # Patch*coordinates of center of gravity for each patch
    # y = torch.rand([10,20,8]) # Batch*patch*channel
    # model_ = HandEncoder(channel=[16,32,64])
    # patch_feature = torch.rand([10,22,16])
    # coordinates_cog = torch.rand([10,22,3])
    # model_(patch_feature, coordinates_cog)

    # model = HandDecoder()
    # feature = torch.rand([10,64])
    # coordinates_cog = torch.rand([10,22,3])
    # model(feature, coordinates_cog)

    # model = PatchDecoder()
    # feature = torch.rand([10,32])
    # coordinates_cog = torch.rand([16,3]) # not timescale
    # model(feature, coordinates_cog)

    model = ContinuousCAE(channel_patch=[3,8,16], channel_hand=[16,32,64])
    inputs = torch.rand([10,1152])
    feature_list = make_input_data(inputs)
    coordinates_cog = torch.rand([10,22,3]) # traindata["CoG_Tactile"]
    feature, output = model(inputs, coordinates_cog)
    # import ipdb; ipdb.set_trace()    

# make model input feature from traindata [100,1152] => 22*100*[16 or 24]*3
def make_input_data(inputs, channel=3):
    # for patch in []:
    tac_list = []
    batch_num = len(inputs)
    count = 0
    for finger_patch in [[16,16,16,16+8],[16,16,16,16+8],[16,16,16,16+8],[16,16,16+8],[16]*7]:
        for num_patch in finger_patch:
            tmp_tac = inputs[:,count:count+num_patch*channel]
            # import ipdb; ipdb.set_trace()
            tmp_tac = tmp_tac.view(batch_num, -1, channel)
            tac_list.append(tmp_tac)
            count += num_patch*channel
            # import ipdb; ipdb.set_trace()
    # cog_list = torch.stack(cog_list,dim=1)
    # handlingData.data[num_object][num_csv]["CoG_Tactile"] = cog_list
    # import ipdb; ipdb.set_trace()
    return tac_list


if __name__=="__main__":
    main()