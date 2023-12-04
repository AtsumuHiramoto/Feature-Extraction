#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import numpy as np
from sklearn.model_selection import train_test_split
import random
import time
import glob
import matplotlib.pyplot as plt
import copy
import re
import pandas as pd
import math
import json
from collections import OrderedDict
import torch
import glob
import pickle
# import ObjectSIzeShapeAnalyzer as osa
# import SensorCoordinatesManager as spm

class DataPreprocessor(object):
    def __init__(self) -> None:
        """
        Parameters
        ----------
        input_data: list
            list of using data type
            e.g. input_data=["tactile", "joint"]
            Args:
                tactile: uSkin sensor's tactile data
                joint: Allegro Hand's joint angle
                torque: Allegro Hand's torque value
                tactile_coordinates: uSkin sensor's 3D Euclidean coordinates
                tactile_coordinates_centroid: Centroid coordinates for each tactile sensor's patch
        """

        self.scaling_df_test = None
        self.scaling_mode_test = None

        self.object_name_list = []

        # About cache
        self.cache_data_dir = "./data_cache/"
        self.cache_data_file = self.cache_data_dir + "data_cache.pkl" # Cache data of self.handling_data
        self.cache_data_info_file = self.cache_data_dir + "data_cache_info.json" # self.load_csv_file_list

        # About name of handling_data columns
        self.finger_name_list = ['Index', 'Middle', 'Little', 'Palm']
        self.patch_name_list = ['IndexTip_TactileB00', 'IndexPhalange_TactileB00', 'IndexPhalange_TactileB01', 'IndexPhalange_TactileB02',
                           'MiddleTip_TactileB00', 'MiddlePhalange_TactileB00', 'MiddlePhalange_TactileB01', 'MiddlePhalange_TactileB02',
                           'LittleTip_TactileB00', 'LittlePhalange_TactileB00', 'LittlePhalange_TactileB01', 'LittlePhalange_TactileB02',
                           'ThumbTip_TactileB00', 'ThumbPhalange_TactileB00', 'ThumbPhalange_TactileB01',
                           'Palm_TactileB00', 'Palm_TactileB01', 'Palm_TactileB02']
        self.joint_name_list = ['JointF0J0', 'JointF0J1', 'JointF0J2', 'JointF0J3', 
                                'JointF1J0', 'JointF1J1', 'JointF1J2', 'JointF1J3', 
                                'JointF2J0', 'JointF2J1', 'JointF2J2', 'JointF2J3', 
                                'JointF3J0', 'JointF3J1', 'JointF3J2', 'JointF3J3']
        self.torque_name_list = ['TorqueF0J0', 'TorqueF0J1', 'TorqueF0J2', 'TorqueF0J3', 
                                'TorqueF1J0', 'TorqueF1J1', 'TorqueF1J2', 'TorqueF1J3', 
                                'TorqueF2J0', 'TorqueF2J1', 'TorqueF2J2', 'TorqueF2J3', 
                                'TorqueF3J0', 'TorqueF3J1', 'TorqueF3J2', 'TorqueF3J3']

    def load_handling_dataset(self, load_dir, skip_timestep=1):
        """
        Function to load dataset.
        if proper cache data is found, then load cache data.
        if proper cache data isn't found, then load csv data and make cache data.

        Parameters
        ----------
        load_dir: str
            Directory path which contains csv data.
            You can use regular expression.
            e.g. load_dir="hoge/*/*/"
        """

        self.load_dir = load_dir
        if type(self.load_dir) is str:
            self.load_csv_file_list = glob.glob(self.load_dir + "*.csv")
        elif type(self.load_dir) is list:
            self.load_csv_file_list = []
            for tmp_load_dir in self.load_dir:
                self.load_csv_file_list += glob.glob(tmp_load_dir + "*.csv")

        self.load_csv_num = len(self.load_csv_file_list)


        if self.check_cache_data()==True:
            self.handling_data = self.load_cache_data()
        else:
            handling_data_df = self.load_csv_data()
            self.handling_data = self.convert_dataframe2tensor(handling_data_df)
            self.make_cache_data()
        
        self.handling_data["data"] = self.handling_data["data"][::skip_timestep]

        return self.handling_data
    
    def load_csv_data(self):
        """
        Function to load all csv data.
        The data is saved in self.handling.data
        """

        if self.load_csv_num==0:
            print("{} doesn't have csv file".format(self.load_dir))
            exit()

        print("Loading starts")
        for csv_id, load_csv_file in enumerate(self.load_csv_file_list):
            print("Loading [{}/{}]: {}".format(csv_id+1, self.load_csv_num, load_csv_file))
            load_df = pd.read_csv(load_csv_file)

            # create new columns
            object_name = load_csv_file.split("/")[-3]
            if object_name not in self.object_name_list:
                self.object_name_list.append(object_name)
            # object_id = len(self.object_name_list) - 1 # object_id starts from 0
            # object_degree = int(load_csv_file.split("/")[-2])
            # filename = load_csv_file.split("/")[-1]
            csv_info_df = pd.DataFrame({"csv_id" : csv_id}, index=load_df.index)
            # csv_info_df = pd.DataFrame({"csv_id" : csv_id, 
            #                        "object_id" : object_id, 
            #                        "orientation_id" : object_degree}, 
            #                        index=load_df.index)
            
            # add csv_info columns
            load_df = pd.concat([csv_info_df, load_df], axis=1)

            # concat csv data
            if csv_id==0:
                handling_data_df = load_df
            else:
                handling_data_df = pd.concat([handling_data_df, load_df])
        print("Loading is completed!")

        return handling_data_df

    def convert_dataframe2tensor(self, handling_data_df):
        """
        Function to convert handling_data from Pandas DataFrame to Pytorch tensor.
        It's very slow to use DataFrame, so this function is important.

        Parameters
        ----------
        handling_data_df: DataFrame
            handling data with DataFrame format
        
        Return
        ------
        handling_data: dictionary
                columns: numpy.ndarray
                    columns of handling_data_df
                data: torch.Tensor
                    handling_data_df.value converted to tensor
        """

        print("Converting from DataFrame to tensor...")
        handling_data = {}
        handling_data["columns"] = handling_data_df.columns.values
        handling_data["data"] = torch.tensor(handling_data_df.values)
        handling_data["load_files"] = self.load_csv_file_list
        print("Converting is completed!")

        return handling_data

    def check_cache_data(self):
        """
        Function to check if cache data exists.
        
        Return
        ------
        True: If cache data is found, and cache data is same as csv.
        False: If cache data isn't found, or cache data isn't same as csv.
        """

        if os.path.isfile(self.cache_data_info_file):
            print("Loading cache information...")
            with open(self.cache_data_info_file) as f:
                cache_csv_list = json.load(f)
            if cache_csv_list==self.load_csv_file_list:
                print("Cache data matched!")
                return True
            else:
                print("Cache data didn't match.")
                return False
        else:
            print("Cache doesn't exist in {}".format(self.cache_data_info_file))
            return False
    
    def load_cache_data(self):
        """
        Function to load cache data.
        The data is saved in self.handling_data.
        """

        print("Loading cache data...")
        # self.handling_data = pd.read_pickle(self.cache_data_file)
        with open(self.cache_data_file, "rb") as f:
            handling_data = pickle.load(f)
        print("Loading is completed!")

        return handling_data
    
    def make_cache_data(self):
        """
        Function to make cache data from loaded csv
        """

        if os.path.isdir(self.cache_data_dir)==False:
            os.mkdir(self.cache_data_dir)

        # save cache_data_info
        with open(self.cache_data_info_file, "w") as f:
            json.dump(self.load_csv_file_list, f, indent=2)
            print("Saved cache information")
        
        # save cache_data
        # self.handling_data.to_pickle(self.cache_data_file)
        with open(self.cache_data_file, "wb") as f:
            pickle.dump(self.handling_data, f)
        print("Saved cache data")

    def add_noise(self, stdev, input_data="tactile"):
        if input_data=="tactile":
            # import ipdb; ipdb.set_trace()
            mask_index = [bool(re.match(".*Tactile.*", s)) for s in self.handling_data["columns"]]
            self.handling_data["data"][:,mask_index] = \
                self.handling_data["data"][:,mask_index] + \
                    torch.normal(mean=0, std=stdev, size=self.handling_data["data"][:,mask_index].shape)
    
    def load_scaling_params(self, scaling_df_path):
        """
        Function to load scaling dataframe to scale test datasets
        """
        df = pd.read_csv(scaling_df_path)
        if df[df.columns[0]][0]=="max":
            self.scaling_mode_test = "normalization"
        if df[df.columns[0]][0]=="mean":
            self.scaling_mode_test = "standardization"
        self.scaling_df_test = df.drop(columns=df.columns[0])

    def filter_tactile(self, filter_range=[-1000,1000]):
        for axis in ["X", "Y", "Z"]:
            hand_1d_column = [bool(re.match(".*Tactile.*{}".format(axis), s)) for s in self.handling_data["columns"]]
            # if axis=="X" or axis=="Y":
            self.handling_data["data"][:, hand_1d_column] = torch.where(self.handling_data["data"][:, hand_1d_column]>filter_range[1], filter_range[1], self.handling_data["data"][:, hand_1d_column])
            self.handling_data["data"][:, hand_1d_column] = torch.where(self.handling_data["data"][:, hand_1d_column]<filter_range[0], filter_range[0], self.handling_data["data"][:, hand_1d_column])
            # import ipdb; ipdb.set_trace()
        return self.handling_data
            # elif axis=="Z":
            #     self.handling_data["data"][:, hand_1d_column] = np.where(self.handling_data["data"][:, hand_1d_column]>1000, 0, self.handling_data["data"][:, hand_1d_column])
            #     self.handling_data["data"][:, hand_1d_column] = np.where(self.handling_data["data"][:, hand_1d_column]<-1000, -1000, self.handling_data["data"][:, hand_1d_column])
    
    def scaling_handling_dataset(self, 
                                 input_data,
                                 output_data,
                                 scaling_mode="normalization", 
                                 scaling_range="patch", 
                                 separate_axis=True, 
                                 separate_joint=True,
                                 tactile_scale=None,
                                 normalization_range=[0.0,1.0]):
        """
        Function to scale data.
        The scaling parameters are saved in self.scaling_df
        
        Parameters
        ----------
        scaling_mode: str
            normalization: Use normalization method
            standardization: Use standardization method
        scaling_range: str
            patch: Scaling for each patches
            hand: Scaling whole hand
        separate_axis: bool
            About tactile sensor's axis
            True: Scaling for each tactile axis (x,y,z)
            False: Scaling whole axis
        separate_joint: bool
            About Allegro Hand's joints
            True: Scaling for each joints
            False: Scaling whole joints
        
        Return
        ------
        self.handling_data: Scaled data
        """

        self.input_data = input_data
        self.output_data = output_data
        self.normalization_range = normalization_range

        # self.scaling_param = {"scaling_mode" : scaling_mode, 
        #                       "scaling_range" : scaling_range, 
        #                       "separate_axis" : separate_axis, 
        #                       "separate_joint" : separate_joint}

        # for Test
        if self.scaling_mode_test is not None:
            scaling_mode = self.scaling_mode_test

        if scaling_mode=="normalization":
            self.scaling_df = pd.DataFrame(columns=self.handling_data["columns"], index=["max", "min"])
        elif scaling_mode=="standardization":
            self.scaling_df = pd.DataFrame(columns=self.handling_data["columns"], index=["mean", "std"])

        input_output_data = self.input_data + self.output_data
        if "tactile" in input_output_data:
            self.scaling_tactile(scaling_mode, scaling_range, separate_axis, tactile_scale)
        if "joint" in input_output_data:
            self.scaling_joint(scaling_mode, separate_joint)
        if "desjoint" in input_output_data:
            self.scaling_desjoint(scaling_mode, separate_joint)
        if "torque" in input_output_data:
            self.scaling_torque(scaling_mode, separate_joint)

        # Under construction
        if "tactile_coordinates" in self.input_data:
            pass
        if "tactile_coordinates_centroid" in self.input_data:
            pass

        return self.handling_data, self.scaling_df

    def scaling_tactile(self, scaling_mode, scaling_range, separate_axis, tactile_scale=None):
        """
        Function to scale tactile data in self.handling_data.
        Use regular expression to search the proper columns
        """

        print("Scaling tactile data...")
        if tactile_scale is not None:
            tactile_column = [bool(re.match(".*Tactile.*", s)) for s in self.handling_data["columns"]]
            if tactile_scale=="sqrt":
                self.handling_data["data"][:, tactile_column] = self.handling_data["data"][:, tactile_column] / torch.sqrt(torch.abs(self.handling_data["data"][:, tactile_column]) + 10e-5)
            elif tactile_scale=="log":
                # import ipdb; ipdb.set_trace()
                self.handling_data["data"][:, tactile_column] = torch.log(torch.abs(self.handling_data["data"][:, tactile_column]) + 1.0) * self.handling_data["data"][:, tactile_column] / torch.abs(self.handling_data["data"][:, tactile_column] + 10e-5)
            # import ipdb; ipdb.set_trace()

        if scaling_range=="patch":
            if separate_axis==True:
                for patch_name in self.patch_name_list:
                    for axis in ["X", "Y", "Z"]:
                        # print("Normalize patch:{} axis:{}".format(patch_name, axis))
                        # Extract columns which matches patch_name
                        patch_1d_column = [bool(re.match("{}.*{}".format(patch_name, axis), s)) for s in self.handling_data["columns"]]
                        if scaling_mode=="normalization":
                            self.normalization(patch_1d_column)
                        elif scaling_mode=="standardization":
                            self.standardization(patch_1d_column)
            elif separate_axis==False:
                patch_3d_column = [bool(re.match("{}.*".format(patch_name), s)) for s in self.handling_data["columns"]]
                if scaling_mode=="normalization":
                    self.normalization(patch_3d_column)
                elif scaling_mode=="standardization":
                    self.standardization(patch_3d_column)
        elif scaling_range=="hand":
            if separate_axis==True:
                for axis in ["X", "Y", "Z"]:
                    hand_1d_column = [bool(re.match(".*Tactile.*{}".format(axis), s)) for s in self.handling_data["columns"]]
                    if scaling_mode=="normalization":
                        self.normalization(hand_1d_column)
                    elif scaling_mode=="standardization":
                        self.standardization(hand_1d_column)
            elif separate_axis==False:
                hand_3d_column = [bool(re.match(".*Tactile.*", s)) for s in self.handling_data["columns"]]
                if scaling_mode=="normalization":
                    self.normalization(hand_3d_column)
                    import ipdb; ipdb.set_trace()
                elif scaling_mode=="standardization":
                    self.standardization(hand_3d_column)
        print("Scaling tactile data is completed!")
    
    def scaling_joint(self, scaling_mode, separate_joint=True):
        """
        Function to scale joint data in self.handling_data.
        Use regular expression to search the proper columns
        """
        
        print("Scaling joint data...")
        if separate_joint==True:
            for joint_name in self.joint_name_list:
                joint_column = [bool(re.match("{}".format(joint_name), s)) for s in self.handling_data["columns"]]
                if scaling_mode=="normalization":
                    self.normalization(joint_column)
                elif scaling_mode=="standardization":
                    self.standardization(joint_column)
        elif separate_joint==False:
            whole_joint_column = [bool(re.match("Joint", s)) for s in self.handling_data["columns"]]
            if scaling_mode=="normalization":
                self.normalization(whole_joint_column)
            elif scaling_mode=="standardization":
                self.standardization(whole_joint_column)
        print("Scaling joint data is completed!")
    
    def scaling_desjoint(self, scaling_mode, separate_joint=True):
        """
        Function to scale desired joint data in self.handling_data.
        Use regular expression to search the proper columns
        """
        
        print("Scaling desjoint data...")
        if separate_joint==True:
            for joint_name in self.joint_name_list:
                desjoint_column = [bool(re.match("{}".format("Des" + joint_name), s)) for s in self.handling_data["columns"]]
                if scaling_mode=="normalization":
                    self.normalization(desjoint_column)
                elif scaling_mode=="standardization":
                    self.standardization(desjoint_column)
        elif separate_joint==False:
            whole_joint_column = [bool(re.match("DesJoint", s)) for s in self.handling_data["columns"]]
            if scaling_mode=="normalization":
                self.normalization(whole_joint_column)
            elif scaling_mode=="standardization":
                self.standardization(whole_joint_column)
        print("Scaling joint data is completed!")

    def scaling_torque(self, scaling_mode, separate_torque=True):
        """
        Function to scale torque data in self.handling_data.
        Use regular expression to search the proper columns
        """
        
        print("Scaling torque data...")
        if separate_torque==True:
            for torque_name in self.torque_name_list:
                torque_column = [bool(re.match("{}".format(torque_name), s)) for s in self.handling_data["columns"]]
                if scaling_mode=="normalization":
                    # import ipdb; ipdb.set_trace()
                    self.normalization(torque_column)
                elif scaling_mode=="standardization":
                    self.standardization(torque_column)
        elif separate_torque==False:
            whole_torque_column = [bool(re.match("Torque", s)) for s in self.handling_data["columns"]]
            # import ipdb; ipdb.set_trace()
            if scaling_mode=="normalization":
                self.normalization(whole_torque_column)
            elif scaling_mode=="standardization":
                self.standardization(whole_torque_column)
        print("Scaling joint data is completed!")

    def normalization(self, target_column):
        """
        Function to normalize data.

        Parameters
        ----------
        target_column: list
            Boolean list.
            True: Normalize the column 
            False Skip the column
        """

        if self.scaling_df_test is None:
            # Normalization process
            df_max = torch.max(self.handling_data["data"][:, target_column])
            df_min = torch.min(self.handling_data["data"][:, target_column])
        else:
            # import ipdb; ipdb.set_trace()
            df_max = self.scaling_df_test.loc[0][target_column].values.reshape(1,-1)
            df_min = self.scaling_df_test.loc[1][target_column].values.reshape(1,-1)
            # import ipdb; ipdb.set_trace()
        # Save scaling parameters
        self.scaling_df.loc["max"][target_column] = df_max
        self.scaling_df.loc["min"][target_column] = df_min

        data_min = self.normalization_range[0]
        data_max = self.normalization_range[1]
        data_range = data_max - data_min
        # self.handling_data["data"][:, target_column] = (self.handling_data["data"][:, target_column] - df_min) / (df_max - df_min)
        self.handling_data["data"][:, target_column] = (((self.handling_data["data"][:, target_column] - df_min) * data_range) / (df_max - df_min)) + data_min

    def standardization(self, target_column):
        """
        Function to standardize data.

        Parameters
        ----------
        target_column: list
            Boolean list.
            True: Standardize the column 
            False Skip the column
        """

        if self.scaling_df_test is None:
            # Standardization process
            df_mean = torch.mean(self.handling_data["data"][:, target_column])
            df_std = torch.std(self.handling_data["data"][:, target_column])
        else:
            df_mean = self.scaling_df_test.loc[0][target_column].values.reshape(1,-1)
            df_std = self.scaling_df_test.loc[1][target_column].values.reshape(1,-1)
        # Save scaling parameters
        self.scaling_df.loc["mean"][target_column] = df_mean
        self.scaling_df.loc["std"][target_column] = df_std
        self.handling_data["data"][:, target_column] = (self.handling_data["data"][:, target_column] - df_mean) / df_std
    
    def split_handling_data(self, split_ratio=[4,1], devide_csv=True, extend_timestep=0):
        """
        Function to make train/test data from self.handling_data
        """

        if devide_csv==True:
            handling_data_list = []
            for i in range(self.load_csv_num):
                csv_mask = (self.handling_data["data"][:,0]==i)
                data = self.handling_data["data"][csv_mask,:]
                data = self.select_input_data(data)
                extend_data = data[-1,:].repeat((extend_timestep,1))
                data = torch.cat([data, extend_data])
                # handling_data_list.append(self.handling_data["data"][csv_mask,:])
                handling_data_list.append(data)

        test_size = split_ratio[1] / sum(split_ratio)
        if test_size > 0:
            train_data, test_data = train_test_split(handling_data_list, test_size=test_size)
            del self.handling_data["data"]
            train_data, train_data_length = self.align_data_length(train_data)
            test_data, test_data_length = self.align_data_length(test_data)
            self.handling_data["train_data"] = train_data
            self.handling_data["train_data_length"] = train_data_length
            self.handling_data["test_data"] = test_data
            self.handling_data["test_data_length"] = test_data_length        
        else:
            train_data = handling_data_list
            test_data = None
            del self.handling_data["data"]
            train_data, train_data_length = self.align_data_length(train_data)
            self.handling_data["train_data"] = train_data
            self.handling_data["train_data_length"] = train_data_length

        return self.handling_data
    
    def align_data_length(self, data):
        data_num = len(data)
        data_length_list = [len(data[i]) for i in range(data_num)]
        max_length = max(data_length_list)
        # import ipdb; ipdb.set_trace()
        aligned_data = []
        for tmp_data in data:
            repeat_dim = max_length - len(tmp_data)
            repeat_data = tmp_data[-1,:].repeat((repeat_dim,1))
            tmp_data = torch.cat([tmp_data, repeat_data])
            aligned_data.append(tmp_data)
        aligned_data = torch.stack(aligned_data)
        # import ipdb; ipdb.set_trace()

        return aligned_data, data_length_list
    
    def select_input_data(self, data):
        """
        Function to select input data and convert dataset format
        """
        data
        return data
    
    def pose_command2label(self):
        #[20, 21, 22, 23]
        #[10,3,4]
        #[30][40-43][50, 51]
        open_pose = [4, 10, 11, 12, 13, 14]
        command_column = [bool(re.match("PoseCommand", s)) for s in self.handling_data["columns"]]
        pose_command = self.handling_data["data"][:, command_column].int()
        # import ipdb; ipdb.set_trace()
        label = torch.zeros_like(pose_command).bool()
        for pose in open_pose:
            label += (pose_command==pose)
        label = label.int()
        # import ipdb; ipdb.set_trace()
        # label = np.where((pose_command==3) + (pose_command==4) + (pose_command==10) + (pose_command==30)\
        #                   + ((pose_command>=20)*(pose_command<=23)) + ((pose_command>=40)*(pose_command<=43))\
        #                   + ((pose_command>=50)*(pose_command<=51)), np.ones_like(pose_command), 
        #                   np.zeros_like(pose_command))
        # self.handling_data["data"][:, command_column] = label
        # import ipdb; ipdb.set_trace()
        add_column_list = []
        if "PoseCommand" not in self.handling_data["columns"]:
            label = np.zeros([len(label), 2])
            add_column_list.append("PoseCommand")
        self.handling_data["data"] = torch.from_numpy(np.concatenate([self.handling_data["data"], label], axis=1))
        
        add_column_list.append("Label")
        # self.handling_data["columns"] = np.concatenate([self.handling_data["columns"], add_column_list])

        # switching_point = np.zeros_like(pose_command)
        # for i in range(len(switching_point)):
        #     if (pose_command[i,0]==3)or(pose_command[i,0]==4)or(pose_command[i,0]==10):
        #         if (pose_command[i+1,0]!=3)and(pose_command[i+1,0]!=4)and(pose_command[i+1,0]!=10):
        #             switching_point[i,0] = pose_command[i,0]
        #             # import ipdb; ipdb.set_trace()
        switching_pose_list = [3,14] # base pose, after opening
        switching_point = np.zeros_like(pose_command)
        for i in range(len(switching_point)):
            for switching_pose in switching_pose_list:
                if (pose_command[i,0]==switching_pose):
                    if (pose_command[i+1,0]!=switching_pose):
                        switching_point[i,0] = pose_command[i,0]
                        # import ipdb; ipdb.set_trace()
        self.handling_data["data"] = torch.from_numpy(np.concatenate([self.handling_data["data"], switching_point], axis=1))
        add_column_list.append("SwitchingPoint")
        self.handling_data["columns"] = np.concatenate([self.handling_data["columns"], add_column_list])
        # import ipdb; ipdb.set_trace()
        return self.handling_data
    
    def trim_label_data(self):
        mask = self.handling_data["data"][:,-1].bool()
        self.handling_data["data"] = self.handling_data["data"][mask,:]
        return self.handling_data

    def handlingDataSplit(handlingData, ratio=[7,3,0]):
        trainData = []
        testData = []
        valData = []
        for num_object in range(8):
            # tmp_trainData = []
            # tmp_testData = []
            # tmp_valData = []
            for num_csv in range(10):
                if num_csv<ratio[0]:
                    # tmp_trainData.append(handlingData.data[num_object][num_csv])
                    trainData.append(handlingData.data[num_object][num_csv])
                elif num_csv<(ratio[0]+ratio[1]):
                    # tmp_testData.append(handlingData.data[num_object][num_csv])
                    testData.append(handlingData.data[num_object][num_csv])
                else:
                    # tmp_valData.append(handlingData.data[num_object][num_csv])
                    valData.append(handlingData.data[num_object][num_csv])
            # trainData.append(tmp_trainData)
            # testData.append(tmp_testData)
            # valData.append(tmp_valData)
        return trainData, testData, valData

    def calcCenterOfGravity(handlingData, L=0):
        for num_object in range(len(handlingData.data)):
            for num_csv in range(len(handlingData.data[num_object])):
                cog_list = []
                count = 0
                for finger_patch in [[16,16,16,16+8],[16,16,16,16+8],[16,16,16,16+8],[16,16,16+8],[16]*7]:
                    for num_patch in finger_patch:
                        tmp_ctac = torch.tensor(handlingData.data[num_object][num_csv]["Coordinates_Tactile"][:,count:count+num_patch*3],
                        dtype=torch.float32)
                        cog_x = tmp_ctac[:,0::3].mean(dim=1)
                        cog_y = tmp_ctac[:,1::3].mean(dim=1)
                        cog_z = tmp_ctac[:,2::3].mean(dim=1)
                        cog_list.append(torch.stack([cog_x,cog_y,cog_z], dim=1))
                        count += num_patch*3
                # import ipdb; ipdb.set_trace()
                cog_list = torch.stack(cog_list,dim=1)
                # Normalization
                if L>0:
                    cog_list = (cog_list - cog_list.min())/(cog_list.max()-cog_list.min())
                    # scaling [-1,1] for positional encoding
                    cog_list_pe = cog_list*2 -1
                    tmp_cog_pe = []
                    for l in range(L):
                        val = 2**l
                        # import ipdb; ipdb.set_trace()
                        tmp_cog_pe.append(torch.sin(val*cog_list_pe*math.pi))
                        tmp_cog_pe.append(torch.cos(val*cog_list_pe*math.pi))
                    # import ipdb; ipdb.set_trace()
                    cog_list_pe = torch.cat(tmp_cog_pe,dim=2)
                    handlingData.data[num_object][num_csv]["CoG_Tactile_pe"] = cog_list_pe

                handlingData.data[num_object][num_csv]["CoG_Tactile"] = cog_list