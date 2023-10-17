import os
# import DataPreProcessor as dpp
import glob
import pandas as pd
import torch
# from torchsummary import summary
from argparse import ArgumentParser
import yaml
from tqdm import tqdm
# from make_dataset import MyDataset
# from Trainer import Train
from autoencoder import AutoEncoder
# from graph_autoencoder_timescale import ContinuousCAE
# from graph_autoencoder_0222 import ContinuousCAE
# from graph_autoencoder_0423 import ContinuousCAE
from utils.data_preproccessor import DataPreprocessor
from utils.make_dataset import MyDataset
from utils.callback import EarlyStopping
from utils.visualizer import Visualizer
from collections import OrderedDict
from layer.lstm import BasicLSTM
from layer.ae import BasicAE
from bptt_trainer import fullBPTTtrainer
from trainer import Trainer
import torch.optim as optim

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
    print("Loading config file...")
    with open(yaml_filepath) as file:
        cfg = yaml.safe_load(file)
    return cfg

def main():
    args = get_option()
    cfg = load_yaml(args.yaml)
    # parameter to load dataset
    if args.mode=="Train":
        load_dir = cfg["file"]["load_dir"]
    if args.mode=="Test":
        load_dir = cfg["test"]["load_dir"]    
    input_data_type = cfg["data"]["input_data"]
    #parameter for scaling
    scaling_mode = cfg["scaling"]["mode"]
    scaling_range = cfg["scaling"]["range"]
    separate_axis = cfg["scaling"]["separate_axis"]
    separate_joint = cfg["scaling"]["separate_joint"]

    model_name = cfg["model"]["model_name"]
    # parameter for positional encoding
    positional_encoding_input = cfg["positional_encoding"]["input_data"]
    positional_encoding_dim = cfg["positional_encoding"]["dimention"]

    split_ratio = cfg["data"]["train_test_val_ratio"]
    devide_csv = cfg["data"]["devide_csv"]

    optimizer_type = cfg["model"]["optimizer"]
    learning_rate = cfg["model"]["learning_rate"]
    tactile_loss = cfg["model"]["tactile_loss"]
    joint_loss = cfg["model"]["joint_loss"]

    dpp = DataPreprocessor(input_data_type)
    handling_data = dpp.load_handling_dataset(load_dir)
    # import ipdb; ipdb.set_trace()
    if args.mode=="Test":
        scaling_df_path = cfg["test"]["scaling_df_path"]
        dpp.load_scaling_params(scaling_df_path)
    handling_data, scaling_df = dpp.scaling_handling_dataset(scaling_mode,
                                                 scaling_range,
                                                 separate_axis,
                                                 separate_joint)
    # scaling paramとae_yamlの値を保存
    # ./weight/{yyyy_mm_dd_hhmmss}/
    # epoch.pth / ccae.yaml / scaling_param.json / loss.png
    # hist など、HandlingDataMaker()で分析関数
    # import ipdb; ipdb.set_trace()

    handling_data = dpp.split_handling_data(split_ratio, devide_csv)
    train_dataset = MyDataset(handling_data, mode="train", input_data=input_data_type)
    if split_ratio[1] > 0: # if you use test data
        test_dataset = MyDataset(handling_data, mode="test", input_data=input_data_type)
    if model_name=="lstm":

        epoch = cfg["model"]["epoch"]
        rec_dim = cfg["model"]["rec_dim"]
        batch_size = cfg["model"]["batch_size"]
        activation = cfg["model"]["activation"]
        seq_num = cfg["model"]["seq_num"]
        model_ae_name = cfg["model"]["model_ae_name"]

        # for train_data in train_loader:
        #     import ipdb; ipdb.set_trace()

        tactile_num = 1104
        joint_num = 16

        if model_ae_name is None:
            model_ae = None
            scaling_df_ae = None
            in_dim = tactile_num + joint_num
            # train_lstm(train_data, test_data)
            model = BasicLSTM(in_dim=in_dim,
                            rec_dim=rec_dim,
                            out_dim=in_dim,
                            activation=activation)
            save_weight_dir = "./output/lstm/"
        else:
            if model_ae_name=="ae":
                model_filepath_ae = cfg["model"]["ae"]["model_filepath"]
                scaling_df_path_ae = cfg["model"]["ae"]["scaling_df_path"]
                scaling_df_ae = pd.read_csv(scaling_df_path_ae)
                scaling_df_ae.index = scaling_df_ae[scaling_df_ae.columns[0]].values
                scaling_df_ae = scaling_df_ae.drop(columns=scaling_df_ae.columns[0])
                # import ipdb; ipdb.set_trace()
                hid_dim = cfg["model"]["ae"]["hid_dim"]
                activation_ae = cfg["model"]["ae"]["activation"]
                model_ae = BasicAE(in_dim=tactile_num,
                          hid_dim=hid_dim,
                          out_dim=tactile_num,
                          activation=activation_ae
                )
                ckpt = torch.load(model_filepath_ae, map_location=torch.device('cpu'))
                model_ae.load_state_dict(ckpt["model_state_dict"])

                in_dim = hid_dim + joint_num
                # train_lstm(train_data, test_data)
                model = BasicLSTM(in_dim=in_dim,
                                rec_dim=rec_dim,
                                out_dim=in_dim,
                                activation=activation)
                save_weight_dir = "./output/ae_lstm/"

        if optimizer_type=="adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type=="radam":
            optimizer = optim.RAdam(model.parameters(), lr=learning_rate)
        else:
            assert False, 'Unknown optimizer name {}. please set Adam or RAdam.'.format(args.optimizer)
        loss_weights = [tactile_loss, joint_loss]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer = fullBPTTtrainer(model, optimizer, loss_weights, model_ae=model_ae, device=device)
        early_stop = EarlyStopping(patience=100000)

        # save_weight_dir = "./weight/lstm/"
        # save_weight_dir = "./output/lstm/"
        if os.path.isdir(save_weight_dir)==False:
            os.makedirs(save_weight_dir)
            os.makedirs(save_weight_dir + "weight/")
            os.makedirs(save_weight_dir + "result/")
        scaling_df.to_csv(save_weight_dir + "scaling_params.csv")

        train_loss_list = []
        test_loss_list = []
        if args.mode=="Train":
            print("Start training!")            
            with tqdm(range(epoch)) as pbar_epoch:
                for epoch in pbar_epoch:
                    if split_ratio[1] > 0:
                        # train and test
                        train_loss = trainer.process_epoch(train_dataset, batch_size=batch_size, seq_num=seq_num)
                        test_loss  = trainer.process_epoch(test_dataset, batch_size=batch_size, seq_num=seq_num, training=False)
                        # writer.add_scalar('Loss/train_loss', train_loss, epoch)
                        # writer.add_scalar('Loss/test_loss',  test_loss,  epoch)

                        # early stop
                        save_ckpt, _ = early_stop(test_loss)

                        if save_ckpt:
                            save_name = save_weight_dir + "weight/lstm_{}.pth".format(epoch)
                            trainer.save(epoch, [train_loss, test_loss], save_name )

                        # print process bar
                        pbar_epoch.set_postfix(OrderedDict(train_loss=train_loss,
                                                            test_loss=test_loss))
                        train_loss_list.append(train_loss)
                        test_loss_list.append(test_loss)
                    else:
                        # train and test
                        train_loss = trainer.process_epoch(train_dataset, batch_size=batch_size, seq_num=seq_num)
                        # writer.add_scalar('Loss/train_loss', train_loss, epoch)

                        save_name = save_weight_dir + "weight/lstm_{}.pth".format(epoch)
                        trainer.save(epoch, [train_loss], save_name )

                        # print process bar
                        pbar_epoch.set_postfix(OrderedDict(train_loss=train_loss))
                        train_loss_list.append(train_loss)

            print("Finished training!")
            # import ipdb; ipdb.set_trace()
            # Save loss image
            v = Visualizer()
            if split_ratio[1] > 0:
                v.save_loss_image(train_loss=train_loss_list,
                                test_loss=test_loss_list,
                                save_dir=save_weight_dir,
                                model_name=model_name,
                                mode="log10")
            else:
                v.save_loss_image(train_loss=train_loss_list,
                                save_dir=save_weight_dir,
                                model_name=model_name,
                                mode="log10")                
        # Save predicted joint
        if args.mode=="Test":
            save_result_dir = save_weight_dir + "result/"
            print("Start joint prediction!") 
            test_model_filepath = cfg["test"]["model_filepath"]
            ckpt = torch.load(test_model_filepath, map_location=torch.device('cpu'))
            model.load_state_dict(ckpt["model_state_dict"])
            trainer.plot_prediction(train_dataset, 
                                    scaling_df=scaling_df, 
                                    scaling_df_ae=scaling_df_ae,
                                    batch_size=batch_size, 
                                    save_dir=save_result_dir,
                                    seq_num=seq_num,
                                    prefix="train")
            if split_ratio[1] > 0:
                trainer.plot_prediction(test_dataset, 
                                        scaling_df=scaling_df, 
                                        scaling_df_ae=scaling_df_ae,
                                        batch_size=batch_size, 
                                        save_dir=save_result_dir,
                                        seq_num=seq_num,
                                        prefix="test")
            print("Finished prediction!")

    if model_name=="ae":


        epoch = cfg["model"]["epoch"]
        hid_dim = cfg["model"]["hid_dim"]
        batch_size = cfg["model"]["batch_size"]
        activation = cfg["model"]["activation"]
        # seq_num = cfg["model"]["seq_num"]

        # for train_data in train_loader:
        #     import ipdb; ipdb.set_trace()

        tactile_num = 1104
        joint_num = 16
        in_dim = tactile_num

        # train_lstm(train_data, test_data)
        model = BasicAE(in_dim=in_dim,
                          hid_dim=hid_dim,
                          out_dim=in_dim,
                          activation=activation)
        if optimizer_type=="adam":
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_type=="radam":
            optimizer = optim.RAdam(model.parameters(), lr=learning_rate)
        else:
            assert False, 'Unknown optimizer name {}. please set Adam or RAdam.'.format(args.optimizer)
        # loss_weights = [tactile_loss, joint_loss]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        trainer = Trainer(model, optimizer, device=device)
        early_stop = EarlyStopping(patience=100000)

        # save_weight_dir = "./weight/lstm/"
        save_weight_dir = "./output/ae/"
        if os.path.isdir(save_weight_dir)==False:
            os.makedirs(save_weight_dir)
            os.makedirs(save_weight_dir + "weight/")
            os.makedirs(save_weight_dir + "result/")
        scaling_df.to_csv(save_weight_dir + "scaling_params.csv")

        train_loss_list = []
        test_loss_list = []
        if args.mode=="Train":
            print("Start training!")            
            with tqdm(range(epoch)) as pbar_epoch:
                for epoch in pbar_epoch:
                    # train and test
                    # import ipdb; ipdb.set_trace()
                    train_loss = trainer.process_epoch(train_dataset, batch_size=batch_size)
                    test_loss  = trainer.process_epoch(test_dataset, batch_size=batch_size, training=False)
                    # writer.add_scalar('Loss/train_loss', train_loss, epoch)
                    # writer.add_scalar('Loss/test_loss',  test_loss,  epoch)

                    # early stop
                    save_ckpt, _ = early_stop(test_loss)

                    if save_ckpt:
                        save_name = save_weight_dir + "weight/ae_{}.pth".format(epoch)
                        trainer.save(epoch, [train_loss, test_loss], save_name )

                    # print process bar
                    pbar_epoch.set_postfix(OrderedDict(train_loss=train_loss,
                                                        test_loss=test_loss))
                    train_loss_list.append(train_loss)
                    test_loss_list.append(test_loss)        
            print("Finished training!")
            # import ipdb; ipdb.set_trace()
            # Save loss image
            v = Visualizer()
            v.save_loss_image(train_loss=train_loss_list,
                            test_loss=test_loss_list,
                            save_dir=save_weight_dir,
                            model_name=model_name,
                            mode="log10")
        # Save predicted joint
        if args.mode=="Test":
            save_result_dir = save_weight_dir + "result/"
            print("Start joint prediction!") 
            test_model_filepath = cfg["test"]["model_filepath"]
            ckpt = torch.load(test_model_filepath, map_location=torch.device('cpu'))
            model.load_state_dict(ckpt["model_state_dict"])
            trainer.plot_prediction(train_dataset, 
                                    scaling_df=scaling_df, 
                                    batch_size=batch_size, 
                                    save_dir=save_result_dir,
                                    prefix="train")
            trainer.plot_prediction(test_dataset, 
                                    scaling_df=scaling_df, 
                                    batch_size=batch_size, 
                                    save_dir=save_result_dir,
                                    prefix="test")
            print("Finished prediction!")
    return 

    if len(positional_encoding_input) > 0:
        # Under construction
        dpp.positional_encoding(positional_encoding_input, positional_encoding_dim)

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