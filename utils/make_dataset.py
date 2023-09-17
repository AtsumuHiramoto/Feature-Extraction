from torch.utils.data import Dataset
import torch
import re

class MyDataset(Dataset):
    # def __init__(self, handling_data, input_data, device, transform=None):
    def __init__(self, handling_data, mode, input_data=[], stdev=0.02):
        super().__init__()
        if mode=="train":
            self.data = handling_data["train_data"]
        elif mode=="test":
            self.data = handling_data["test_data"]
        self.columns = handling_data["columns"]
        self.file_names = handling_data["load_files"]

        self.input_data = input_data
        # self.device = device
        # self.csv_num = len(self.data)
        self.stdev = stdev

    def __getitem__(self, index):
        x_data = {}
        y_data = {}
        if "joint" in self.input_data:
            joint_mask = [bool(re.match("Joint", s)) for s in self.columns]
            joint_data = self.data[index][:, joint_mask]
            joint_data = joint_data.float()
            # joint_data = torch.t(joint_data).float()
            # add gaussian noise to input data
            x_data["joint"] = joint_data + torch.normal(mean=0, std=self.stdev, size=joint_data.shape)
            y_data["joint"] = joint_data
        if "tactile" in self.input_data:
            tactile_mask = [bool(re.match(".*Tactile", s)) for s in self.columns]
            tactile_data = self.data[index][:, tactile_mask]
            tactile_data = tactile_data.float()
            # tactile_data = torch.t(tactile_data).float()
            x_data["tactile"] = tactile_data
            # x_data["tactile"] = tactile_data + torch.normal(mean=0, std=self.stdev, size=tactile_data.shape)
            y_data["tactile"] = tactile_data
        file_name = self.file_names[index]
        return [x_data, y_data, file_name]

    def __len__(self) -> int:
        return len(self.data)

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