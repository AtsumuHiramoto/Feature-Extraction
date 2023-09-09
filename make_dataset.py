from torch.utils.data import Dataset
import torch

class MyDataset(Dataset):
    def __init__(self, data, inputType, device, transform=None) -> None:
        super().__init__()
        self.data = data
        self.inputType = inputType
        self.device = device
        self.csv_num = len(self.data)
        self.timestep = self.data[0][self.inputType[0]].shape[0]

    def __getitem__(self, index):
        data_dict = dict()
        for key in self.inputType:
            data_dict[key] = torch.tensor(self.data[int(index/self.timestep)][key][index%self.timestep,:]).to(self.device)
            # data_dict[key] = torch.tensor(self.data[int(index/self.timestep)][key][index%self.timestep,:])
            # data_dict[key] = data_dict[key].clone().detach().to(self.device)
        return data_dict
        # return data_dict, int(index/self.timestep), index%self.timestep

    def __len__(self) -> int:
        self.length = self.csv_num * self.timestep
        return self.length