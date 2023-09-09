import DataPreProcessor as dpp
import glob
import torch
from Trainer import Train
# from graph_autoencoder_timescale import ContinuousCAE
# from graph_autoencoder_0222 import ContinuousCAE
from graph_autoencoder_0423 import ContinuousCAE
from autoencoder import AutoEncoder
from torch.utils.data import Dataset
from torchsummary import summary
from argparse import ArgumentParser
import yaml
from make_dataset import MyDataset

def get_option():
    argparser = ArgumentParser()
    argparser.add_argument('-y', '--yaml', type=str,
                           default="./config/default.yaml",
                           help='Path of hyper parameter YAML file')
    argparser.add_argument('-m', '--mode', type=str,
                           default="Train",
                           help='Train or Test')
    return argparser.parse_args()

def load_yaml(yaml_filepath):
    with open(yaml_filepath) as file:
        cfg = yaml.safe_load(file)
    return cfg

def main():
    args = get_option()
    cfg = load_yaml(args.yaml)
    loadDirs = cfg["filepath"]["loadDirs"]
    loadDirs = glob.glob(loadDirs)
    # import ipdb; ipdb.set_trace()
    inputType = cfg["data"]["inputType"]
    outputType = cfg["data"]["outputType"]
    split_ratio = cfg["data"]["split_ratio"]
    additionalInputType = cfg["data"]["additionalInputType"]
    encode_pe_flag = cfg["model"]["PositionalEncoding"]["encode_pe_flag"]
    decode_pe_flag = cfg["model"]["PositionalEncoding"]["decode_pe_flag"]
    L = cfg["model"]["PositionalEncoding"]["L"]
    scaling_method = cfg["model"]["scaling_method"]
    handlingData = dpp.LoadHandlingData(loadDirs,inputType, outputType) # data (8,10,[300,1152] etc)
    # import ipdb; ipdb.set_trace()
    scaling_params = dpp.ScalingHandlingData(handlingData, mode='DataLimit', scaling_method=scaling_method)
    # import ipdb; ipdb.set_trace()
    if "Coordinates_Tactile" in inputType:
        if decode_pe_flag:
            dpp.calcCenterOfGravity(handlingData, L=L)
        else:
            dpp.calcCenterOfGravity(handlingData, L=0)
    # import ipdb; ipdb.set_trace()
    for additional_input in cfg["data"]["additionalInputType"]:
        inputType.append(additional_input)
    trainData, testData, valData = dpp.handlingDataSplit(handlingData, ratio=split_ratio)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dataset = MyDataset(trainData, inputType, device)
    test_dataset = MyDataset(testData, inputType, device)

    # import ipdb; ipdb.set_trace()
    jointName = handlingData.columnName[0:16]
    tactileName = handlingData.columnName[16:1168]
    labelName = handlingData.columnName[1168:1174]
    jointCoordinatesName = handlingData.columnName[1174:1222]
    tactileCoordinatesName = handlingData.columnName[1222:]
    # saveCoordinatesImage_(loadDirs,tactileCoordinatesName)
    # saveCoordinatesImage_j(loadDirs,jointCoordinatesName)
    channel_patch = cfg["model"]["channel_patch"]
    channel_hand = cfg["model"]["channel_hand"]
    # model = ContinuousCAE(channel_patch=channel_patch, channel_hand=channel_hand, decode_pe_flag=decode_pe_flag, cfg=cfg).to(device)
    model = AutoEncoder()
    # summary(model)
    # import ipdb; ipdb.set_trace()
    # x = 
    Train(model, train_dataset, test_dataset, decode_pe_flag=decode_pe_flag, cfg=cfg)

    # saveimg=True
    # if saveimg:
    #     saveCoordinatesImage(handlingData,tactileCoordinatesName)

if __name__=="__main__":
    main()