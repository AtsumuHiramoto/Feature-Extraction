# fork from eipl
import torch
import torch.nn as nn
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from ah_tactile_player import AHTactilePlayer

    
class fullBPTTtrainer:
    """
    Helper class to train recurrent neural networks with numpy sequences

    Args:
        traindata (np.array): list of np.array. First diemension should be time steps
        model (torch.nn.Module): rnn model
        optimizer (torch.optim): optimizer
        input_param (float): input parameter of sequential generation. 1.0 means open mode.
    """

    def __init__(self,
                input_data,
                output_data,
                model,
                optimizer,
                loss_weights=[1.0, 1.0, 1.0, 1.0], # [tactile, joint, torque, label]
                model_ae=None,
                device='cpu',
                tactile_scale=None,
                loss_constraint=None):
        
        self.input_data = input_data
        self.output_data = output_data
        self.device = device
        self.optimizer = optimizer
        self.loss_weights = loss_weights
        self.model = model.to(self.device)
        self.model_ae = model_ae
        if self.model_ae is not None:
            self.model_ae = self.model_ae.to(self.device)
            for param in self.model_ae.parameters():
                param.requires_grad = False
        self.tactile_scale = tactile_scale
        self.loss_constraint = loss_constraint # Kase's method

    def save(self, epoch, loss, savename):
        if len(loss) == 2:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        #'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_loss': loss[0],
                        'test_loss': loss[1],
                        }, savename)
        else:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        #'optimizer_state_dict': self.optimizer.state_dict(),
                        'train_loss': loss[0],
                        }, savename)

    def split_dataset(self, data, batch_size=100):
        # import ipdb; ipdb.set_trace()
        batch_num = math.ceil(data.shape[0] / batch_size)
        # batch_num = math.ceil(data.shape[1] / batch_size)
        data_list = []
        for i in range(batch_num):
            batch_from = i * batch_size
            batch_to = (i + 1) * batch_size
            split_data = data[batch_from:batch_to, :]
            if split_data.shape[0] < batch_size:
                copy_dim = batch_size-split_data.shape[0]
                copied_data = split_data[-1,:].repeat((copy_dim,1))
                split_data = torch.cat([split_data, copied_data])
            data_list.append(split_data)
        # import ipdb; ipdb.set_trace()
        return data_list
    
    def process_epoch(self, dataset, batch_size, seq_num=1, training=True):
        # import ipdb; ipdb.set_trace()
        if not training:
            self.model.eval()
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total_loss = 0.0
        for n_batch, (x_data, y_data, data_length, file_name) in enumerate(data_loader):
        # for data in data_loader:
            # import ipdb; ipdb.set_trace()
            # sequence_num = x_data["tactile"].shape[1]
            # x_tac = self.split_dataset(x_data["tactile"].to(self.device), batch_size=batch_size)
            # x_joint = self.split_dataset(x_data["joint"].to(self.device), batch_size=batch_size)
            x_tac = x_data["tactile"].to(self.device)
            y_tac = y_data["tactile"].to(self.device)
            x_joint = x_data["joint"].to(self.device)
            if "joint" in self.output_data:
                y_joint = y_data["joint"].to(self.device)
            elif "desjoint" in self.output_data:
                y_joint = y_data["desjoint"].to(self.device)
            if "torque" in self.input_data:
                x_torque = x_data["torque"].to(self.device)
                y_torque = y_data["torque"].to(self.device)
                yp_list = []
            if "label" in self.output_data:
                y_label = y_data["label"].float().to(self.device)
                yl_list = []
                pose_command = x_data["pose"].int()
                switching_point = x_data["switching"].int()
            state = None
            yt_list, yj_list, state_list = [], [], []
            # T = seq_num
            T = x_tac.shape[1]
            # import ipdb; ipdb.set_trace()
            if self.model_ae is None:
                for t in range(T-seq_num):
                    if "torque" in self.input_data:
                        if "label" in self.output_data:
                            _yt_hat, _yj_hat, _yp_hat, _yl_hat, state = self.model(x_tac[:,t], x_joint[:,t], x_torque[:,t], state=state)
                            yl_list.append(_yl_hat)
                        else:
                            _yt_hat, _yj_hat, _yp_hat, state = self.model(x_tac[:,t], x_joint[:,t], x_torque[:,t], state=state)
                        yp_list.append(_yp_hat)
                    else:
                        if "label" in self.output_data:
                            _yt_hat, _yj_hat, _yl_hat, state = self.model(x_tac[:,t], x_joint[:,t], state=state)
                            yl_list.append(_yl_hat)
                        else:
                            _yt_hat, _yj_hat, state = self.model(x_tac[:,t], x_joint[:,t], state=state)
                    yt_list.append(_yt_hat)
                    yj_list.append(_yj_hat)
                    state_list.append(state[0])
            else:
                # import ipdb; ipdb.set_trace()
                y_hidden = self.model_ae.encoder(x_tac)
                y_tac = y_hidden
                # import ipdb; ipdb.set_trace()
                for t in range(T-seq_num):
                    yh_hat = self.model_ae.encoder(x_tac[:,t])
                    if "torque" in self.input_data:
                        if "label" in self.output_data:
                            _yh_hat, _yj_hat, _yp_hat, _yl_hat, state = self.model(yh_hat, x_joint[:,t], x_torque[:,t], state=state)
                            yl_list.append(_yl_hat)
                        else:
                            _yh_hat, _yj_hat, _yp_hat, state = self.model(yh_hat, x_joint[:,t], x_torque[:,t], state=state)
                        yp_list.append(_yp_hat)
                    else:
                        if "label" in self.output_data:
                            _yh_hat, _yj_hat, _yl_hat, state = self.model(yh_hat, x_joint[:,t], state=state)
                            yl_list.append(_yl_hat)
                        else:
                            _yh_hat, _yj_hat, state = self.model(yh_hat, x_joint[:,t], state=state)
                    # _yt_hat = self.model_ae.decoder(_yh_hat)
                    _yt_hat = _yh_hat
                    yt_list.append(_yt_hat)
                    yj_list.append(_yj_hat)
                    state_list.append(state[0])
            
            yt_hat = torch.stack(yt_list).permute(1,0,2)
            yj_hat = torch.stack(yj_list).permute(1,0,2)
            state_list = torch.stack(state_list).permute(1,0,2)
            if "torque" in self.input_data:
                yp_hat = torch.stack(yp_list).permute(1,0,2)
            if "label" in self.output_data:
                yl_hat = torch.stack(yl_list).permute(1,0,2)

            # calculate loss using actual data length
            # train only open timestep
            if "thumb" in self.input_data:  
                for i in range(len(yt_hat)):
                    mask_len = len(yt_hat[i,:data_length[i]])
                    loss_mask = y_data["label"][i,:mask_len,0].bool()
                    if i==0:
                        loss = self.loss_weights[0]*nn.MSELoss()(yt_hat[i,:data_length[i]][loss_mask], y_tac[i,seq_num:data_length[i]+seq_num][loss_mask])\
                            + self.loss_weights[1]*nn.MSELoss()(yj_hat[i,:data_length[i]][loss_mask], y_joint[i,seq_num:data_length[i]+seq_num][loss_mask])
                    else:
                        loss += self.loss_weights[0]*nn.MSELoss()(yt_hat[i,:data_length[i]][loss_mask], y_tac[i,seq_num:data_length[i]+seq_num][loss_mask])\
                            + self.loss_weights[1]*nn.MSELoss()(yj_hat[i,:data_length[i]][loss_mask], y_joint[i,seq_num:data_length[i]+seq_num][loss_mask])
                    if "torque" in self.input_data:
                        loss += self.loss_weights[2]*nn.MSELoss()(yp_hat[i,:data_length[i]][loss_mask], y_torque[i,seq_num:data_length[i]+seq_num][loss_mask])
                    if "label" in self.output_data:
                        loss += self.loss_weights[3]*nn.MSELoss()(yl_hat[i,:data_length[i]][loss_mask], y_label[i,seq_num:data_length[i]+seq_num][loss_mask])
            # train all timestep
            else:
                for i in range(len(yt_hat)):
                    if i==0:
                        loss = self.loss_weights[0]*nn.MSELoss()(yt_hat[i,:data_length[i]], y_tac[i,seq_num:data_length[i]+seq_num])\
                            + self.loss_weights[1]*nn.MSELoss()(yj_hat[i,:data_length[i]], y_joint[i,seq_num:data_length[i]+seq_num])
                    else:
                        loss += self.loss_weights[0]*nn.MSELoss()(yt_hat[i,:data_length[i]], y_tac[i,seq_num:data_length[i]+seq_num])\
                            + self.loss_weights[1]*nn.MSELoss()(yj_hat[i,:data_length[i]], y_joint[i,seq_num:data_length[i]+seq_num])
                    if "torque" in self.input_data:
                        loss += self.loss_weights[2]*nn.MSELoss()(yp_hat[i,:data_length[i]], y_torque[i,seq_num:data_length[i]+seq_num])
                    if "label" in self.output_data:
                        loss += self.loss_weights[3]*nn.MSELoss()(yl_hat[i,:data_length[i]], y_label[i,seq_num:data_length[i]+seq_num])
                    if self.loss_constraint is not None:
                        tmp_switching_point = switching_point[i,:-1].flatten()
                        switching_id = 3
                        # mask = (tmp_switching_point==switching_id).flatten()
                        switching_timestep_list = [t for t, x in enumerate(tmp_switching_point) if x==switching_id]
                        if len(switching_timestep_list) > 1:
                            for j in range(switching_timestep_list-1):
                                loss += self.loss_constraint*nn.MSELoss()(state_list[i, switching_timestep_list[j]],
                                                                          state_list[i, switching_timestep_list[j+1]])
                    
                    # if self.loss_constraint is not None:
                    #     # import ipdb; ipdb.set_trace()
                    #     tmp_switching_point = switching_point[i,:-1]
                    #     for switching_id in [3, 4, 10]:
                    #         mask = (tmp_switching_point==switching_id).flatten()
                    #         if mask.sum() > 1:
                    #             loss += self.loss_constraint*torch.sum(torch.var(state_list[i, mask], dim=0))

            # loss = self.loss_weights[0]*nn.MSELoss()(yt_hat, y_tac[:,1:]) + self.loss_weights[1]*nn.MSELoss()(yj_hat, y_joint[:,1:])
            # yt_hat = torch.stack(yt_list).permute(2,0,1)[:,:,0]
            # yt_hat = torch.permute(torch.stack(yt_list), (1,0,2,3,4) )
            # yj_hat = torch.stack(yj_list).permute(2,0,1)[:,:,0]
            # yj_hat = torch.permute(torch.stack(yj_list), (1,0,2) )
            # loss = self.loss_weights[0]*nn.MSELoss()(yt_hat, y_tac[:,1:] ) + self.loss_weights[1]*nn.MSELoss()(yj_hat, y_joint[:,1:] )
            total_loss += loss.item()

            if training:
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()

        return total_loss / (n_batch+1)

    def plot_prediction(self, dataset, scaling_df, scaling_df_ae, batch_size, save_dir, seq_num=1, prefix=""):
        self.model.eval()
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total_loss = 0.0
        for n_batch, (x_data, y_data, data_length, file_name) in enumerate(data_loader):
            sequence_num = x_data["tactile"].shape[1]
            x_tac = x_data["tactile"].to(self.device)
            y_tac = y_data["tactile"].to("cpu").detach().numpy()
            x_joint = x_data["joint"].to(self.device)
            if "joint" in self.output_data:
                y_joint = y_data["joint"].to("cpu").detach().numpy()
            elif "desjoint" in self.output_data:
                y_joint = y_data["desjoint"].to("cpu").detach().numpy()
            if "torque" in self.input_data:
                x_torque = x_data["torque"].to(self.device)
                y_torque = y_data["torque"].to("cpu").detach().numpy()
                yp_list = []
            if "label" in self.output_data:
                y_label = y_data["label"].to("cpu").detach().numpy()
                yl_list = []
                switching_point = x_data["switching"].int().numpy()
            else:
                switching_point = None
            state = None
            states = []
            yt_list, yj_list, yh_list = [], [], []
            T = x_tac.shape[1]
            if self.model_ae is None:
                for t in range(T-seq_num):
                    _yt_hat, _yj_hat, state = self.model(x_tac[:,t], x_joint[:,t], state=state)
                    yt_list.append(_yt_hat)
                    yj_list.append(_yj_hat)
                    states.append(state[0])
            else:
                # import ipdb; ipdb.set_trace()
                y_hidden = self.model_ae.encoder(x_tac)
                # import ipdb; ipdb.set_trace()
                for t in range(T-seq_num):
                    yh_hat = self.model_ae.encoder(x_tac[:,t])
                    if "torque" in self.input_data:
                        if "label" in self.output_data:
                            _yh_hat, _yj_hat, _yp_hat, _yl_hat, state = self.model(yh_hat, x_joint[:,t], x_torque[:,t], state=state)
                            yl_list.append(_yl_hat)
                        else:
                            _yh_hat, _yj_hat, _yp_hat, state = self.model(yh_hat, x_joint[:,t], x_torque[:,t], state=state)
                        yp_list.append(_yp_hat)
                    else:
                        if "label" in self.output_data:
                            _yh_hat, _yj_hat, _yl_hat, state = self.model(yh_hat, x_joint[:,t], state=state)
                            yl_list.append(_yl_hat)
                        else:
                            _yh_hat, _yj_hat, state = self.model(yh_hat, x_joint[:,t], state=state)
                    _yt_hat = self.model_ae.decoder(_yh_hat)
                    yh_list.append(_yh_hat)
                    yt_list.append(_yt_hat)
                    yj_list.append(_yj_hat)
                    states.append(state[0])
                yh_hat = torch.stack(yh_list).permute(1,0,2).to("cpu").detach().numpy()
                y_hidden = y_hidden.to("cpu").detach().numpy()
            yt_hat = torch.stack(yt_list).permute(1,0,2).to("cpu").detach().numpy()
            yj_hat = torch.stack(yj_list).permute(1,0,2).to("cpu").detach().numpy()
            if "torque" in self.input_data:
                yp_hat = torch.stack(yp_list).permute(1,0,2).to("cpu").detach().numpy()
            if "label" in self.output_data:
                yl_hat = torch.stack(yl_list).permute(1,0,2).to("cpu").detach().numpy()

            # import ipdb; ipdb.set_trace()
            # color_list = ["mediumblue", "blue", "dodgerblue", "cyan",
            #               "darkgreen", "green", "lime", "greenyellow",
            #               "indigo", "blueviolet", "mediumpurple", "plum",
            #               "firebrick", "red", "tomato", "lightcoral"]
            color_list = ["mediumblue", "blue", "dodgerblue", "cyan",
                          "grey", "green", "lime", "greenyellow",
                          "olive", "blueviolet", "mediumpurple", "plum",
                          "red", "orange", "yellow", "mediumvioletred"]
            if scaling_df_ae is None:
                y_tac = self.rescaling_data(y_tac, scaling_df, data_type="tactile")
                yt_hat = self.rescaling_data(yt_hat, scaling_df, data_type="tactile")
            else:
                y_tac = self.rescaling_data(y_tac, scaling_df_ae, data_type="tactile")
                yt_hat = self.rescaling_data(yt_hat, scaling_df_ae, data_type="tactile")                          
            original_csv_column = scaling_df.columns.values[1:]

            if self.model_ae is not None:
                for i in range(len(y_hidden)):
                    plt.figure(figsize=(15,5))
                    if "thumb" in self.input_data:  
                        mask_len = len(yt_hat[i,:data_length[i]])
                        loss_mask = y_data["label"][i,:mask_len,0]
                        plt.plot(range(len(loss_mask)), loss_mask, "--", color="r")
                    # plt.plot(range(y_joint.shape[1]), y_joint[i, seq_num:], ":")
                    # plt.plot(range(yj_hat.shape[1]), yj_hat[i], "-")
                    # import ipdb; ipdb.set_trace()
                    # for j in range(y_hidden.shape[2]):
                    for j in range(y_hidden.shape[2]):
                        # plt.plot(range(data_length[i]-seq_num), y_hidden[i, seq_num:data_length[i]][:,j], ":", color=color_list[j])
                        # plt.plot(range(data_length[i]-seq_num), yh_hat[i, :data_length[i]-seq_num][:,j], "-", color=color_list[j])
                        plt.plot(range(data_length[i]-seq_num), y_hidden[i, seq_num:data_length[i],j], ":")
                        plt.plot(range(data_length[i]-seq_num), yh_hat[i, :data_length[i]-seq_num,j], "-")
                    # plt.show()
                    save_title = file_name[i].split("/")[-1].replace(".csv", "")
                    plt.title(save_title)
                    save_file_name = save_dir + prefix + "_tacF_" + save_title
                    plt.savefig(save_file_name + ".png")
                    plt.close()

            # for i in range(len(y_tac)):
            #     y_tac_df = self.convert_array2pandas(y_tac[i], original_csv_column)
            #     yt_hat_df = self.convert_array2pandas(yt_hat[i], original_csv_column)
            #     # import ipdb; ipdb.set_trace()
            #     save_title = save_dir + file_name[i].split("/")[-1].replace(".csv", "")
            #     AHTactilePlayer([y_tac_df[seq_num:int(data_length[i])], yt_hat_df[:int(data_length[i])-seq_num]],
            #                     5, 0.6, save_title)

            y_joint = self.rescaling_data(y_joint, scaling_df, data_type="joint")
            yj_hat = self.rescaling_data(yj_hat, scaling_df, data_type="joint")
            for i in range(len(y_joint)):
                plt.figure(figsize=(15,5))
                if "thumb" in self.input_data:  
                        mask_len = len(yt_hat[i,:data_length[i]])
                        loss_mask = y_data["label"][i,:mask_len,0]
                        plt.plot(range(len(loss_mask)), loss_mask, "--", color="r")
                # plt.plot(range(y_joint.shape[1]), y_joint[i, seq_num:], ":")
                # plt.plot(range(yj_hat.shape[1]), yj_hat[i], "-")
                for j in range(y_joint.shape[2]):
                    plt.plot(range(data_length[i]-seq_num), y_joint[i, seq_num:data_length[i]][:,j], ":", color=color_list[j])
                    plt.plot(range(data_length[i]-seq_num), yj_hat[i, :data_length[i]-seq_num][:,j], "-", color=color_list[j])
                # plt.show()
                save_title = file_name[i].split("/")[-1].replace(".csv", "")
                plt.title(save_title)
                save_file_name = save_dir + prefix + "_joint_" + save_title
                plt.savefig(save_file_name + ".png")
                plt.close()
            
            if "torque" in self.input_data:
                y_torque = self.rescaling_data(y_torque, scaling_df, data_type="torque")
                yp_hat = self.rescaling_data(yp_hat, scaling_df, data_type="torque")
                for i in range(len(y_torque)):
                    plt.figure(figsize=(15,5))
                    if "thumb" in self.input_data:  
                        mask_len = len(yt_hat[i,:data_length[i]])
                        loss_mask = y_data["label"][i,:mask_len,0]
                        plt.plot(range(len(loss_mask)), loss_mask, "--", color="r")
                    # plt.plot(range(y_joint.shape[1]), y_joint[i, seq_num:], ":")
                    # plt.plot(range(yj_hat.shape[1]), yj_hat[i], "-")
                    for j in range(y_torque.shape[2]):
                        plt.plot(range(data_length[i]-seq_num), y_torque[i, seq_num:data_length[i]][:,j], ":", color=color_list[j])
                        plt.plot(range(data_length[i]-seq_num), yp_hat[i, :data_length[i]-seq_num][:,j], "-", color=color_list[j])
                    # plt.show()
                    save_title = file_name[i].split("/")[-1].replace(".csv", "")
                    plt.title(save_title)
                    save_file_name = save_dir + prefix + "_torque_" + save_title
                    plt.savefig(save_file_name + ".png")
                    plt.close()
            
            if "label" in self.output_data:
                for i in range(len(y_label)):
                    plt.figure(figsize=(15,5))
                    plt.plot(range(data_length[i]-seq_num), y_label[i, seq_num:data_length[i]], ":", color="b")
                    plt.plot(range(data_length[i]-seq_num), yl_hat[i, :data_length[i]-seq_num], "-", color="r")
                    # plt.show()
                    save_title = file_name[i].split("/")[-1].replace(".csv", "")
                    plt.title(save_title)
                    save_file_name = save_dir + prefix + "_label_" + save_title
                    plt.savefig(save_file_name + ".png")
                    plt.close()

            # import ipdb; ipdb.set_trace()
            color_list_pca = ["blue", "cyan",
                          "green", "greenyellow",
                          "blueviolet", "plum",
                          "red", "orange"]
            self.plot_pca(states, save_file_name + ".gif", color_list=color_list_pca)
            with open(save_file_name + ".txt", "w") as f:
                f.write(str(file_name)+str(color_list_pca))
        return

    def closed_loop(self, dataset, scaling_df, scaling_df_ae, batch_size, save_dir, seq_num=1, prefix=""):
        self.model.eval()
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
        total_loss = 0.0
        for n_batch, (x_data, y_data, data_length, file_name) in enumerate(data_loader):
            sequence_num = x_data["tactile"].shape[1]
            x_tac = x_data["tactile"].to(self.device)
            y_tac = y_data["tactile"].to("cpu").detach().numpy()
            x_joint = x_data["joint"].to(self.device)
            if "joint" in self.output_data:
                y_joint = y_data["joint"].to("cpu").detach().numpy()
            elif "desjoint" in self.output_data:
                y_joint = y_data["desjoint"].to("cpu").detach().numpy()
            if "torque" in self.input_data:
                x_torque = x_data["torque"].to(self.device)
                y_torque = y_data["torque"].to("cpu").detach().numpy()
                yp_list = []
            state = None
            states = []
            yt_list, yj_list, yh_list = [], [], []
            T = x_tac.shape[1]
            if self.model_ae is None:
                for t in range(T-seq_num):
                    _yt_hat, _yj_hat, state = self.model(x_tac[:,t], x_joint[:,t], state=state)
                    yt_list.append(_yt_hat)
                    yj_list.append(_yj_hat)
                    states.append(state[0])
            else:
                # import ipdb; ipdb.set_trace()
                y_hidden = self.model_ae.encoder(x_tac)
                # import ipdb; ipdb.set_trace()
                for t in range(T-seq_num):
                    yh_hat = self.model_ae.encoder(x_tac[:,t])
                    if "torque" in self.input_data:
                        if "label" in self.output_data:
                            _yh_hat, _yj_hat, _yp_hat, _yl_hat, state = self.model(yh_hat, x_joint[:,t], x_torque[:,t], state=state)
                        else:
                            _yh_hat, _yj_hat, _yp_hat, state = self.model(yh_hat, x_joint[:,t], x_torque[:,t], state=state)
                        yp_list.append(_yp_hat)
                    else:
                        _yh_hat, _yj_hat, state = self.model(yh_hat, x_joint[:,t], state=state)
                    _yt_hat = self.model_ae.decoder(_yh_hat)
                    yh_list.append(_yh_hat)
                    yt_list.append(_yt_hat)
                    yj_list.append(_yj_hat)
                    states.append(state[0])
                for t in range(T-seq_num):
                    if "torque" in self.input_data:
                        _yh_hat, _yj_hat, _yp_hat, state = self.model(_yh_hat, _yj_hat, _yp_hat, state=state)
                        yp_list.append(_yp_hat)
                    else:
                        _yh_hat, _yj_hat, state = self.model(_yh_hat, _yj_hat, state=state)
                    yh_list.append(_yh_hat)
                    yt_list.append(_yt_hat)
                    yj_list.append(_yj_hat)
                    states.append(state[0])
                yh_hat = torch.stack(yh_list).permute(1,0,2).to("cpu").detach().numpy()
                y_hidden = y_hidden.to("cpu").detach().numpy()
            yt_hat = torch.stack(yt_list).permute(1,0,2).to("cpu").detach().numpy()
            yj_hat = torch.stack(yj_list).permute(1,0,2).to("cpu").detach().numpy()
            if "torque" in self.input_data:
                yp_hat = torch.stack(yp_list).permute(1,0,2).to("cpu").detach().numpy()

            # import ipdb; ipdb.set_trace()
            # color_list = ["mediumblue", "blue", "dodgerblue", "cyan",
            #               "darkgreen", "green", "lime", "greenyellow",
            #               "indigo", "blueviolet", "mediumpurple", "plum",
            #               "firebrick", "red", "tomato", "lightcoral"]
            color_list = ["mediumblue", "blue", "dodgerblue", "cyan",
                          "grey", "green", "lime", "greenyellow",
                          "olive", "blueviolet", "mediumpurple", "plum",
                          "red", "orange", "yellow", "mediumvioletred"]
            if scaling_df_ae is None:
                y_tac = self.rescaling_data(y_tac, scaling_df, data_type="tactile")
                yt_hat = self.rescaling_data(yt_hat, scaling_df, data_type="tactile")
            else:
                y_tac = self.rescaling_data(y_tac, scaling_df_ae, data_type="tactile")
                yt_hat = self.rescaling_data(yt_hat, scaling_df_ae, data_type="tactile")                          
            original_csv_column = scaling_df.columns.values[1:]

            y_joint = self.rescaling_data(y_joint, scaling_df, data_type="joint")
            yj_hat = self.rescaling_data(yj_hat, scaling_df, data_type="joint")
            for i in range(len(y_joint)):
                plt.figure(figsize=(15,5))
                # plt.plot(range(y_joint.shape[1]), y_joint[i, seq_num:], ":")
                # plt.plot(range(yj_hat.shape[1]), yj_hat[i], "-")
                for j in range(16):
                    plt.plot(range(data_length[i]-seq_num), y_joint[i, seq_num:data_length[i]][:,j], ":", color=color_list[j])
                    plt.plot(range(2*(data_length[i]-seq_num)), yj_hat[i, :2*(data_length[i]-seq_num)][:,j], "-", color=color_list[j])
                # plt.show()
                save_title = file_name[i].split("/")[-1].replace(".csv", "")
                plt.title(save_title)
                save_file_name = save_dir + prefix + "_closed_joint_" + save_title
                plt.savefig(save_file_name + ".png")
        return

    def plot_joint(self, y_joint, yj_hat, data_length, seq_num, save_dir, file_name, prefix):
        for i in range(len(y_joint)):
            plt.figure(figsize=(15,5))
            # plt.plot(range(y_joint.shape[1]), y_joint[i, seq_num:], ":")
            # plt.plot(range(yj_hat.shape[1]), yj_hat[i], "-")
            for j in range(16):
                plt.plot(range(data_length[i]-seq_num), y_joint[i, seq_num:data_length[i]][:,j], ":", color=color_list[j])
                plt.plot(range(data_length[i]-seq_num), yj_hat[i, :data_length[i]-seq_num][:,j], "-", color=color_list[j])
            # plt.show()
            save_title = file_name[i].split("/")[-1].replace(".csv", "")
            plt.title(save_title)
            save_file_name = save_dir + prefix + "_" + save_title
            plt.savefig(save_file_name + ".png")

    def plot_pca(self, states, save_file_name, color_list=[]):
        import matplotlib.animation as anim
        from sklearn.decomposition import PCA
        import numpy as np
        states = torch.stack(states).permute(1,0,2)
        states = states.to("cpu").detach().numpy()
        N,T,D  = states.shape
        states = states.reshape(-1,D)
        # loop_ct = float(360)/T
        speed = 100
        loop_ct = float(360)/speed
        pca_dim = 3
        pca     = PCA(n_components=pca_dim).fit(states)
        pca_val = pca.transform(states)
        # Reshape the states from [-1, pca_dim] to [N,T,pca_dim] to
        # visualize each state as a 3D scatter.
        pca_val = pca_val.reshape( N, T, pca_dim )

        fig = plt.figure(dpi=60)
        ax = fig.add_subplot(projection='3d')

        def anim_update(i):
            ax.cla()
            angle = int(loop_ct * i)
            ax.view_init(30, angle)

            # c_list = ['C0','C1','C2','C3','C4']
            if len(color_list) < N:
                c_list = ["C{}".format(s) for s in range(N)]
            else:
                c_list = color_list[:N]
            for n, color in enumerate(c_list):
                ax.scatter( pca_val[n,1:,0], pca_val[n,1:,1], pca_val[n,1:,2], alpha=0.5, color=color, s=3.0 )

            ax.scatter( pca_val[n,0,0], pca_val[n,0,1], pca_val[n,0,2], color='k', s=30.0 )
            for n, color in enumerate(c_list):
                ax.scatter( pca_val[n,-1,0], pca_val[n,-1,1], pca_val[n,-1,2], color=color, s=30.0 )
            pca_ratio = pca.explained_variance_ratio_ * 100
            ax.set_xlabel('PC1 ({:.1f}%)'.format(pca_ratio[0]) )
            ax.set_ylabel('PC2 ({:.1f}%)'.format(pca_ratio[1]) )
            ax.set_zlabel('PC3 ({:.1f}%)'.format(pca_ratio[2]) )

        # ani = anim.FuncAnimation(fig, anim_update, interval=int(np.ceil(T/10)), frames=T)
        ani = anim.FuncAnimation(fig, anim_update, interval=int(np.ceil(speed*8/10)), frames=speed)
        # ani.save( './output/PCA_{}.gif'.format(save_file_name) )
        ani.save(save_file_name)


    def convert_array2pandas(self, array, column, data_type="tactile"):
        # import ipdb; ipdb.set_trace()
        data = np.zeros([len(array), len(column)])
        if data_type=="tactile":
            tac_index_st = np.where(column=="IndexTip_TactileB00C00X")[0][0]
            tac_index_en = np.where(column=='Palm_TactileB02C23Z')[0][0] + 1
            data[:, tac_index_st:tac_index_en] = array
        df = pd.DataFrame(data=data, columns=column)
        # import ipdb; ipdb.set_trace()
        return df    

    def rescaling_data(self, data, scaling_df, data_type="joint"):
        if "max" in scaling_df.index:
            mode = "normalization"
        elif "mean" in scaling_df.index:
            mode = "standardization"
        else:
            assert False, "scaling_df is invalid."
        if data_type=="joint":
            if "thumb" in self.input_data:
                scaling_param = scaling_df.filter(regex="^JointF3").values
            else:
                scaling_param = scaling_df.filter(regex="^Joint").values # Jointから始まる列
            if mode=="normalization":
                rescaled_data = data * (scaling_param[0] - scaling_param[1]) + scaling_param[1]
            if mode=="standardization":
                rescaled_data = data * scaling_param[1] + scaling_param[0]
        if data_type=="tactile":
            if "thumb" in self.input_data:
                scaling_param = scaling_df.filter(regex="Thumb.*Tactile").values            
            else:
                scaling_param = scaling_df.filter(regex="Tactile").values # Tactileを含む列
            if mode=="normalization":
                rescaled_data = data * (scaling_param[0] - scaling_param[1]) + scaling_param[1]
            if mode=="standardization":
                rescaled_data = data * scaling_param[1] + scaling_param[0]    
            if self.tactile_scale is not None:
                if self.tactile_scale=="sqrt":
                    rescaled_data = rescaled_data * np.abs(rescaled_data)
                if self.tactile_scale=="log":
                    rescaled_data = math.e ** (np.abs(rescaled_data)) * rescaled_data / np.abs(rescaled_data) - 1.0
        if data_type=="torque":
            if "thumb" in self.input_data:
                scaling_param = scaling_df.filter(regex="^TorqueF3").values
            else:
                scaling_param = scaling_df.filter(regex="^Torque").values # Torqueから始まる列
            if mode=="normalization":
                rescaled_data = data * (scaling_param[0] - scaling_param[1]) + scaling_param[1]
            if mode=="standardization":
                rescaled_data = data * scaling_param[1] + scaling_param[0]

        return rescaled_data
            