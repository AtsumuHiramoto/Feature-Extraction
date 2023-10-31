from torch.utils.data import Dataset
import torch
import re

class MyDataset(Dataset):
    # def __init__(self, handling_data, input_data, device, transform=None):
    def __init__(self, handling_data, mode, input_data=[], output_data=[], 
                 stdev_tactile=0.0, stdev_joint=0.02):
        super().__init__()
        if mode=="train":
            data = handling_data["train_data"]
            self.data_length = handling_data["train_data_length"]
        elif mode=="test":
            data = handling_data["test_data"]
            self.data_length = handling_data["test_data_length"]
        self.dataset_num = len(data)
        columns = handling_data["columns"]
        joint_mask = [bool(re.match("Joint", s)) for s in columns]
        self.joint_data = data[:,:,joint_mask].float()
        desjoint_mask = [bool(re.match("DesJoint", s)) for s in columns]
        self.desjoint_data = data[:,:,desjoint_mask].float()
        tactile_mask = [bool(re.match(".*Tactile", s)) for s in columns]
        self.tactile_data = data[:,:,tactile_mask].float()
        self.file_names = handling_data["load_files"]
        # import ipdb; ipdb.set_trace()
        self.input_data = input_data
        self.output_data = output_data
        # self.device = device
        # self.csv_num = len(self.data)
        self.stdev_tactile = stdev_tactile
        self.stdev_joint = stdev_joint

    def __getitem__(self, index):
        x_data = {}
        y_data = {}
        if "joint" in self.input_data:
            # joint_data = torch.t(joint_data).float()
            # add gaussian noise to input data
            # import ipdb; ipdb.set_trace()
            if self.stdev_joint > 0:
                x_data["joint"] = self.joint_data[index] + torch.normal(mean=0, std=self.stdev_joint, size=self.joint_data[index].shape)
            else:
                x_data["joint"] = self.joint_data[index]
            y_data["joint"] = self.joint_data[index]
        if "tactile" in self.input_data:
            # tactile_data = torch.t(tactile_data).float()
            if self.stdev_tactile > 0:
                x_data["tactile"] = self.tactile_data[index] + torch.normal(mean=0, std=self.stdev_tactile, size=self.tactile_data[index].shape)
            else:
                x_data["tactile"] = self.tactile_data[index]
            # x_data["tactile"] = tactile_data + torch.normal(mean=0, std=self.stdev, size=tactile_data.shape)
            y_data["tactile"] = self.tactile_data[index]
        if "desjoint" in self.output_data:
            y_data["desjoint"] = self.desjoint_data[index]
        data_length = self.data_length[index]
        file_name = self.file_names[index]
        return [x_data, y_data, data_length, file_name]

    def __len__(self) -> int:
        return self.dataset_num

class TimeSeriesDataSet(Dataset):
    """
    Fork from eipl

    Args:
        feats (np.array):  Set the image features.
        joints (np.array): Set the joint angles.
        minmax (float, optional):  Set normalization range, default is [0.1,0.9].
    """
    def __init__( self,
                  feats,
                  joints,
                  minmax=[0.1, 0.9],
                  stdev=0.02):

        self.stdev  = stdev
        self.feats  = torch.from_numpy(feats).float()
        self.joints = torch.from_numpy(joints).float()

    def __len__(self):
        return len(self.feats)

    def __getitem__(self, idx):
        
        # normalization and convert numpy array to torch tensor
        y_feat  = self.feats[idx]
        y_joint = self.joints[idx]
        y_data  = torch.concat( (y_feat, y_joint), axis=-1)

        # apply gaussian noise to joint angles and image features
        x_feat  = self.feats[idx]  + torch.normal(mean=0, std=self.stdev, size=y_feat.shape)
        x_joint = self.joints[idx] + torch.normal(mean=0, std=self.stdev, size=y_joint.shape)

        x_data = torch.concat( (x_feat, x_joint), axis=-1)

        return [x_data, y_data]