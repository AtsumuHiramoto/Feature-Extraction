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

    def scaling_handling_dataset(self, 
                                 input_data,
                                 output_data,
                                 scaling_mode="normalization", 
                                 scaling_range="patch", 
                                 separate_axis=True, 
                                 separate_joint=True):
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
            self.scaling_tactile(scaling_mode, scaling_range, separate_axis)
        if "joint" in input_output_data:
            self.scaling_joint(scaling_mode, separate_joint)
        if "desjoint" in input_output_data:
            self.scaling_desjoint(scaling_mode, separate_joint)

        # Under construction
        if "tactile_coordinates" in self.input_data:
            pass
        if "tactile_coordinates_centroid" in self.input_data:
            pass

        return self.handling_data, self.scaling_df

    def scaling_tactile(self, scaling_mode, scaling_range, separate_axis):
        """
        Function to scale tactile data in self.handling_data.
        Use regular expression to search the proper columns
        """

        print("Scaling tactile data...")
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
            df_max = self.scaling_df_test.loc[0][target_column].values.reshape(1,-1)
            df_min = self.scaling_df_test.loc[1][target_column].values.reshape(1,-1)
            # import ipdb; ipdb.set_trace()
        # Save scaling parameters
        self.scaling_df.loc["max"][target_column] = df_max
        self.scaling_df.loc["min"][target_column] = df_min
        self.handling_data["data"][:, target_column] = (self.handling_data["data"][:, target_column] - df_min) / (df_max - df_min)

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

    # def make_train_test_data(self, split_ratio=[4,1], devide_csv=True):
    #     """
    #     Function to make train/test/validation data from self.handling_data
    #     At first, split data into train/test/validation.
    #     Next, split data into each input.
    #     """

    #     handling_data_list = self.split_handling_data(split_ratio, devide_csv)
    #     train_csv_num = split_ratio[0] / sum(split_ratio)
    #     test_csv_num = round(self.load_csv_num * (split_ratio[1] / sum(split_ratio)))
    #     valid_csv_num = self.load_csv_num - train_csv_num - test_csv_num


    #     return

    def split_handling_data(self, split_ratio=[4,1], devide_csv=True):
        """
        Function to make train/test data from self.handling_data
        """

        if devide_csv==True:
            handling_data_list = []
            for i in range(self.load_csv_num):
                csv_mask = (self.handling_data["data"][:,0]==i)
                data = self.handling_data["data"][csv_mask,:]
                data = self.select_input_data(data)
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
        

#===============================================================================
# Hand Parameter
#===============================================================================
# Limit of Sensor Data
# モータ角LIMIT
LIMIT_JOINT = np.array([[-28.80, 27.66], [-12.19, 94.20], [-10.47, 98.00], [-13.80, 93.55],
                        [-28.80, 27.66], [-12.19, 94.20], [-10.47, 98.00], [-13.80, 93.55],
                        [-28.80, 27.66], [-12.19, 94.20], [-10.47, 98.00], [-13.80, 93.55],
                        [ 25.63,  80.94], [-7.03, 61.10], [-11.62, 94.69], [ -9.90, 99.28]]) # , [0, 9500]])
LIMIT_DJOINT = np.array([[-28.80, 27.66], [-12.19, 94.20], [-10.47, 98.00], [-13.80, 93.55],
                        [-28.80, 27.66], [-12.19, 94.20], [-10.47, 98.00], [-13.80, 93.55],
                        [-28.80, 27.66], [-12.19, 94.20], [-10.47, 98.00], [-13.80, 93.55],
                        [ 25.63,  80.94], [-7.03, 61.10], [-11.62, 94.69], [ -9.90, 99.28]]) # , [0, 9500]])
LIMIT_TORQUE = np.array([[-28.80, 27.66], [-12.19, 94.20], [-10.47, 98.00], [-13.80, 93.55],
                        [-28.80, 27.66], [-12.19, 94.20], [-10.47, 98.00], [-13.80, 93.55],
                        [-28.80, 27.66], [-12.19, 94.20], [-10.47, 98.00], [-13.80, 93.55],
                        [ 25.63,  80.94], [-7.03, 61.10], [-11.62, 94.69], [ -9.90, 99.28]])  #Caution: This is not limit Torque
# 6軸力覚センサLIMIT
# 拇指，示指共通
LIMIT_SIXAXIS = np.array([[-500.00, 500.00]])
# 物体サイズLIMIT
LIMIT_SIZE = np.array([[0.01, 0.07]])
# 物体形状情報LIMIT
LIMIT_SHAPE = np.array([[-1, 1]])
# LIMIT_SHAPE = np.array([[-1, 1]]) # Curvature
# 画像センサLIMIT
LIMIT_IMAGE = np.array([[0, 255.0]])
IMAGE_SIZE = 96
# 触覚センサLIMIT
LIMIT_TACTILE = np.array([[-10000.0, 25000.0]]) #Need to change the range of scale by x, y, z
LIMIT_TACTILE_X = np.array([[0.0, 64000.0]]) #Need to change the range of scale by x, y, z
LIMIT_TACTILE_Y = np.array([[0.0, 64000.0]]) #Need to change the range of scale by x, y, z
LIMIT_TACTILE_Z = np.array([[-10000.0, 25000.0]]) #Need to change the range of scale by x, y, z
# Threshold
DROP_FORCE = 4  # 6軸合力がこの値を下回ったら対象物落下とみなす閾値
#===============================================================================
# Class
#===============================================================================
class CHandlingData(object):
    def __init__(self, data=[], Objectlabel=[], fileNames=[], columnName=[], RANGE=None):
        self.data = data
        self.LIMIT = {"JOINT":LIMIT_JOINT, \
              "TORQUE":LIMIT_TORQUE, \
              "FORCE":LIMIT_SIXAXIS, \
              "DJOINT" : LIMIT_DJOINT, \
              "IMAGE" : LIMIT_IMAGE, \
              "TACTILE" : LIMIT_TACTILE, \
              "TACTILE......X" : LIMIT_TACTILE_X, \
              "TACTILE......Y" : LIMIT_TACTILE_Y, \
              "TACTILE......Z" : LIMIT_TACTILE_Z, \
              "SIZE":LIMIT_SIZE, \
              "SHAPE":LIMIT_SHAPE}
        self.RANGE = RANGE
        self.columnName = columnName
        self.fileNames = fileNames
        self.Objectlabel = Objectlabel

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

#author:hiramoto
#Add outputType
def LoadHandlingData(loadDirs,inputType, outputType, raw=None):     ## flag:1 = raw data(column : Index16_Tactile, Index8_Tactile...)
    import os
    import pickle
    script_dir = os.path.abspath(os.path.dirname(__file__)) #pythonファイルのパスを返す /home/handling_team/hiramoto/fnn_generate/Utils
    script_dir += "/data_cache"
    cache_info = {
        'loadDirs': loadDirs,
        'inputType': inputType,
        'outputType': outputType
    }

    print("(dpp LoadHandlingData) Script directory:", script_dir) #/home/handling_team/hiramoto/fnn_generate/Utils
    print("(dpp LoadHandlingData) Data load conf:\n" + json.dumps(cache_info, indent=2)) #8 csv dirs, Torque

    if os.path.isfile(script_dir+'/data_cache_info.json') and\
            os.path.isfile(script_dir+'/data_cache'):
        with open(script_dir +
                  '/data_cache_info.json', 'r') as json_file:
            json_info = json.load(json_file)

            print("Cached glob pattern   :",
                  json_info['loadDirs'], json_info['inputType'], json_info['outputType']) #/home/handling_team/hiramoto/Data/isobe_data_JOINTTorque_Coordinates/mayo/success/ etc, Torque
            print("Specified glob pattern:",
                  cache_info['loadDirs'], cache_info['inputType'], cache_info['outputType']) #/home/handling_team/hiramoto/Data/isobe_data_JOINTTorque_Coordinates/mayo/success/ etc, Torque

            if json_info['loadDirs'] == cache_info['loadDirs'] and\
                    json_info['inputType'] == cache_info['inputType'] and\
                        json_info['outputType'] == cache_info['outputType']:
                print("Loading cached data...")
                with open(script_dir +
                          '/data_cache', 'rb') as cache_file:
                    handling_data = pickle.load(cache_file) #<DataPreProcessor.CHandlingData object at 0x7f24001bb2e8> <-- CHandlingData(HD, Objectlabel, FN, INPUT, ALL_RANGE)
                    #print("cashe:", handling_data)
                    return handling_data
            else:
                print("Cache is not used, " +
                      "because configuration was changed")

    #Cacheが一致しなかった場合↓

    print('Loading csv files...')

    print('\n')
    print('------------------------------')
    print('| Load Handling Data...      |')
    print('------------------------------')
    HD = []
    FN = []
    Objectlabel = []
    input_outputType = []
    input_outputType = [i for i in inputType]
    for i in outputType:
        if (i in inputType)==False:
            input_outputType.append(i)
    for loadDir in loadDirs:
        loadDir_sep = loadDir.split('/')
        #print(loadDir_sep)
        #Objectlabel.append(loadDir_sep[-1].split("\\")[-3])
        # import ipdb; ipdb.set_trace()
        Objectlabel.append(loadDir_sep[-3]) #mayo etc
        # import ipdb; ipdb.set_trace()
        # CSVファイルの読み込み
        handlingData, INPUT, FN = DataIO.LoadFile(loadDir)
        #print(handlingData, INPUT, FN) #handlingData...all csv file data, INPUT...csv header ['JointF0J0' 'JointF0J1' 'JointF0J2' ... ], FN...csv file name ['_reshaped_smoothcut_20201203_180707_allegro_mayo_11.csv', ... ]

        #files = handlingData.axes[0]
        #for i, file in enumerate(files):
        #    print('  [%d]:%s' % (i, file))
        #FN.append(list(map(str,files)))
        new_handlingData = []
        dic = {}

        for f, key in enumerate(handlingData):
            INPUT_DATA = OrderedDict()
            ALL_RANGE = OrderedDict()
            index = 0
            last_index = 0
            # adapt to length of time steps in each csv
            #handlingData_temp = handlingData.values[f]
            #print(f)
            #print("handlingData:", handlingData[f])
            handlingData_temp = handlingData[f].values
            handlingData_temp = pd.DataFrame(handlingData_temp)
            handlingData_temp = handlingData_temp.dropna()
            handlingData_temp = handlingData_temp.values
            #inputType: ['Torque']
            for i in range(len(input_outputType)):
                flag = 0
                # for j in range(len(INPUT)):
                #     if "Tactile" in INPUT[j]:
                #         print("True")
                # import ipdb; ipdb.set_trace()
                for j in range(len(INPUT)):
                    if ((raw == 1) and ((input_outputType[i] == "TACTILE") or (input_outputType[i] == "Tactile") or (input_outputType[i] == "tactile"))):
                        if "Tactile" in INPUT[j]:
                            # import ipdb; ipdb.set_trace()
                            data = handlingData_temp[:,j]
                            index +=1
                            if flag == 0:
                                DATA = data
                                flag = 1
                            else:
                                DATA = np.c_[DATA, data]
                    else:
                        # 文字列の先頭でパターンがマッチするかどうかを判定
                        if re.match(input_outputType[i], INPUT[j], re.IGNORECASE):
                            data = handlingData_temp[:,j]
                            index +=1
                            if flag == 0:
                                DATA = data
                                flag = 1
                            else:
                                DATA = np.c_[DATA, data]
                if flag != 0:
                    INPUT_DATA[input_outputType[i]] = DATA
                    RANGE = list(range(last_index, index))
                    ALL_RANGE[input_outputType[i]] = RANGE
                last_index = index
            new_handlingData.append(INPUT_DATA)
        handlingData  = new_handlingData

        if  'SIZE' in input_outputType:
            print('------------------------------')
            print('| Calculate Object Size...   |')
            print('------------------------------')
#             objectSize, objectSize_ftd = CalculateObjectSize(handlingData.values, contactCenterPos)
            objectSize = osa.CalculateObjectSize_Simple(handlingData)
# 物体サイズを行列に付加
# タクタイルを利用したサイズ推定結果を使う場合
#             dic = {}
#             for i, key in enumerate(handlingData):
#                 handlingData[key]['Size'] = objectSize[i]
#                 handlingData[key]['SizeFtd'] = objectSize_ftd[i]
#                 print objectSize[i]
            for i in range(len(handlingData)):
                handlingData[i]['SIZE'] = objectSize[i][np.newaxis].T
                Range_List = list(ALL_RANGE.values()) # OrderdDict() does not have indexing functions
                ALL_RANGE['SIZE'] = list(range(Range_List[0][-1], Range_List[0][-1]+1))

        if  'SHAPE' in input_outputType:
            print('------------------------------')
            print('| Append Shape info...       |')
            print('------------------------------')
            if 'Sphere' in loadDir:
                print( '>> Shape is [Sphere]')
                handlingData = osa.AppendObjectShapeInformation_simple(handlingData, ALL_RANGE,'Sphere')
            elif 'Cylinder' in loadDir:
                print('>> Shape is [Cylinder]')
                handlingData = osa.AppendObjectShapeInformation_simple(handlingData, ALL_RANGE,'Cylinder')
            else:
                print('>> Shape is [Unknown]')
                handlingData = osa.AppendObjectShapeInformation_simple(handlingData, ALL_RANGE,'Unknown')
            print(np.array(handlingData[0]['JOINT']).shape)
            print(np.array(handlingData[0]['TACTILE']).shape)
            print(np.array(handlingData[0]['SIZE']).shape)
            print(np.array(handlingData[0]['SHAPE']).shape)
        HD.append(handlingData)
    print('Shape & Size', Objectlabel)
    print(ALL_RANGE.values())
    print(np.array(HD).shape) #[8,10]
    print(INPUT, ALL_RANGE)
    print(HD[0][0]) #[('Torque', array([[ 0.02291117, ...]), ('JOINT', array([[-1.45097261e-02, ...])])

    ret_data = CHandlingData(HD, Objectlabel, FN, INPUT, ALL_RANGE)

    with open(script_dir+'/data_cache', 'wb') as f:
        print("Caching data...")
        pickle.dump(ret_data, f)

    with open(script_dir +
              '/data_cache_info.json', 'w') as json_file:
        json.dump(cache_info, json_file)

    print('------------------------------')
    print('| Complete...                |')
    print('------------------------------')
    return ret_data

def Nomalization(data, indataRange, outdataRange):
    data = ( data - indataRange[0] ) / ( indataRange[1] - indataRange[0])
    data = data * (outdataRange[1] - outdataRange[0] ) + outdataRange[0]
    return data

def ImageProcessor(loadDirs):
    print('------------------------------')
    print('| Load Image Data...         |')
    print('------------------------------')
    handlingobjects = [None]*len(loadDirs)
    for i, loadDir in enumerate(loadDirs):
        print(loadDir + "../im*")
        handlingobjects[i] = glob.glob(loadDir + "/../im*")
    DATA = []
    for i in range(len(handlingobjects)):
        data = []
        for (j,oneobject) in enumerate(handlingobjects[i]):
            IMG = glob.glob(oneobject +'/*.png')
            img = []
            for k in IMG:
                img.append(np.asarray(Image.open(k).convert('L').resize((IMAGE_SIZE,IMAGE_SIZE))).astype(np.float32))
            img= np.array(img).reshape(np.array(img).shape[0],np.array(img).shape[1]*np.array(img).shape[2])
            print(np.array(img).shape)
            img = Nomalization(np.array(img).astype(np.float32),[0,255],[0,255])
            data.append(img)
#         print('IMAGE DATA SHAPE1')
#         print(np.array(data).shape)
        DATA.append(np.array(data))
    print('IMAGE DATA SHAPE')
    print(np.array(DATA).shape)
    print('------------------------------')
    print('| Complete...                |')
    print('------------------------------')
    return DATA

def DownSamplingImage(Images, timeInterval=10):
    print(timeInterval)
#     timeInterval = int(timeInterval * 120 / 400) # Number of Images / Number of Time steps
    rate = 400 / 120 / timeInterval
    print('------------------------------------')
    print('| Down Sampling Image...           |')
    print('------------------------------------')
    print(('| TimeInterval: %3d               |' % timeInterval))
    print('------------------------------------')
    IndexNum = int(len(Images[0][0]) * rate)
    # print(rate)
    # print(IndexNum)
    # print(len(Images))
    Index = random.sample(range(len(Images[0][0])),IndexNum)
    Index = np.sort(Index)
    new_Images = []
    for s in range(len(Images)):
        new_Image_csv = []
        Image_csv = Images[s]
        for t in range(len(Image_csv)):
            new_Image_t = []
            Image_t = Image_csv[t]
            for i in Index:
                Image_pix = Image_t[i]
                new_Image_t.append(Image_pix)
            new_Image_csv.append(new_Image_t)
        new_Images.append(new_Image_csv)
#     for s in range(len(Images)):
#         l = Images[s].shape[1]
#         Images[s] = (Images[s][:, list(range(0, l, timeInterval)), :])
    # print(np.array(new_Images).shape)
    return new_Images

def ConcatHandlingImage(HandlingData, Images):
    print('-------------------------------------------------------------------')
    print('| Concatenating Inputs (HandlingData & Images)...                 |')
    print('-------------------------------------------------------------------')
    HANDLINGDATAOBJECT = []
    for i in range(len(HandlingData.data)):# size shape
        handlingdatacsv = HandlingData.data[i]
        imagescsv = Images[i]
        HANDLINGDATACSV = []
        for j in range(len(handlingdatacsv)): # CSV
            handlingdatatime = np.hstack([handlingdatacsv[j],imagescsv[j]]) # time inputs
            HANDLINGDATACSV.append(np.array(handlingdatatime))
        HANDLINGDATAOBJECT.append(np.array(HANDLINGDATACSV))
    HandlingData.data = np.array(HANDLINGDATAOBJECT)
    print('----ImageData Shape----')
    print(np.array(Images).shape)
    print('----ConcatenatedData Shape----')
    print(np.array(HandlingData.data).shape)
    print('------------------------------------------------------------------')
    return HandlingData

def CheckLimit(HandlingData):
    for s in range(len(HandlingData.data)):
        for t in range(len(HandlingData.data[s])):
            for key, val in HandlingData.RANGE.items():
                # Round Data
                if key == 'TIME':
                    pass
                else:
                    if key == 'JOINT':
                        LIMIT = HandlingData.LIMIT['JOINT']
                    if key == 'TORQUE':
                        LIMIT = HandlingData.LIMIT['TORQUE']
                    else:
                        pass
#                             LIMIT = np.tile(HandlingData.LIMIT[key], (len(HandlingData.RANGE[key]), 1))
                for k in range(len(HandlingData.data[s][t][key])):
                    if key == 'TORQUE':
                        pass
                    elif key == 'JOINT':
                        for j in range(len(HandlingData.data[s][t][key][k])):
                            HandlingData.data[s][t][key][k][j] = math.degrees(HandlingData.data[s][t][key][k][j])
                    else:
                        HandlingData.data[s][t][key][k][j] = HandlingData.data[s][t][key][k][j]
                        if HandlingData.data[s][t][key][k][j] < LIMIT[j, 0]:
                            HandlingData.data[s][t][key][k][j] = LIMIT[j, 0]
                        if HandlingData.data[s][t][key][k][j] > LIMIT[j, 1]:
                            HandlingData.data[s][t][key][k][j] = LIMIT[j, 1]

def TactileZeroCalibration(HandlingData):
    print('---------------------------------------------------')
    print('|  Zero calibrating tactile values...           |')
    print('---------------------------------------------------')
    for s in range(len(HandlingData.data)):
        for t in range(len(HandlingData.data[s])):
            for key, val in HandlingData.RANGE.items():
                # Round Data
                if key == 'TACTILE':
                    for i in range(np.array(HandlingData.data[s][t][key]).shape[1]):
                        if i % 3 == 2:
                            HandlingData.data[s][t][key][: , i] = HandlingData.data[s][t][key][: , i] - 13000
                        else:
                            HandlingData.data[s][t][key][: , i] = HandlingData.data[s][t][key][: , i] - 32700

def LowPassFilter(HandlingData):
    print('-----------------------------------')
    print('| Low pass filtering...           |')
    print('-----------------------------------')
    for s in range(len(HandlingData.data)):
        for t in range(len(HandlingData.data[s])):
            for key, val in HandlingData.RANGE.items():
                # Round Data
                if key == 'TACTILE':
                    flag = 0
                    for i in range(np.array(HandlingData.data[s][t][key]).shape[1]):
                        # データのパラメータ
                        Num = len(HandlingData.data[s][t][key])            # サンプル数
                        dt = 1       # サンプリング間隔
                        fc = 0.25           # カットオフ周波数
                        fn = 1/(0.01*2*dt)                   # ナイキスト周波数
                        time = np.arange(0, Num*dt, dt) # np.arange(0, N*dt, dt) # 時間軸
                        freq = np.linspace(0, 1.0/dt, Num) # 周波数軸
                        # パラメータ設定
                        fp = 10                          # 通過域端周波数[Hz]
                        fs = 15                          # 阻止域端周波数[Hz]
                        gpass = 1                       # 通過域最大損失量[dB]
                        gstop = 40                      # 阻止域最小減衰量[dB]
                        # 正規化
                        Wp = fp/fn
                        Ws = fs/fn
                        # 時間信号（周波数5の正弦波 + 周波数40の正弦波）の生成
                        f = copy.deepcopy(HandlingData.data[s][t][key][: , i])
                        # ローパスフィルタで波形整形
                        # バターワースフィルタ
                        N, Wn = signal.buttord(Wp, Ws, gpass, gstop)
                        b1, a1 = signal.butter(N, Wn, "low")
                        y1 = signal.filtfilt(b1, a1, f)
                        # 第一種チェビシェフフィルタ
                        N, Wn = signal.cheb1ord(Wp, Ws, gpass, gstop)
                        b2, a2 = signal.cheby1(N, gpass, Wn, "low")
                        y2 = signal.filtfilt(b2, a2, f)
                        # 第二種チェビシェフフィルタ
                        N, Wn = signal.cheb2ord(Wp, Ws, gpass, gstop)
                        b3, a3 = signal.cheby2(N, gstop, Wn, "low")
                        y3 = signal.filtfilt(b3, a3, f)
                        # 楕円フィルタ
                        N, Wn = signal.ellipord(Wp, Ws, gpass, gstop)
                        b4, a4 = signal.ellip(N, gpass, gstop, Wn, "low")
                        y4 = signal.filtfilt(b4, a4, f)
                        # ベッセルフィルタ
                        N = 4
                        b5, a5 = signal.bessel(N, Ws, "low")
                        y5 = signal.filtfilt(b5, a5, f)
                        # FIR フィルタ
                        a6 = 1
                        numtaps = Num
                        b6 = signal.firwin(numtaps, Wp, window="hann")
                        y6 = signal.lfilter(b6, a6, f)
                        delay = (numtaps-1)/2*dt

                        # 高速フーリエ変換（周波数信号に変換）
                        F = np.fft.fft(f)
                        # 正規化 + 交流成分2倍
                        F = F/(Num/2)
                        F[0] = F[0]/2
                        # 配列Fをコピー
                        F2 = F.copy()
                        # ローパスフィルタ処理（カットオフ周波数を超える帯域の周波数信号を0にする）
                        F2[(freq > fc)] = 0
                        # 高速逆フーリエ変換（時間信号に戻す）
                        f2 = np.fft.ifft(F2)
                        # 振幅を元のスケールに戻す
                        f2 = np.real(f2*Num)
                        HandlingData.data[s][t][key][: , i] = f2

                        if flag == 0:
                            fS = f
                            FS = F
                            F2S = F2
                            flag = 1
                        else:
                            fS = np.c_[fS, f]
                            FS = np.c_[FS, F]
                            F2S = np.c_[F2S, F2]
                    # グラフ表示
#                     plt.figure()
# #                     plt.rcParams['font.family'] = 'Times New Roman'
# #                     plt.rcParams['font.size'] = 17
#                     # 時間信号（元）
#                     plt.subplot(221)
#                     plt.plot(time, fS, label='f(n)')
#                     plt.xlabel("Time", fontsize=20)
#                     plt.ylabel("Signal", fontsize=20)
#                     plt.title("Original Trajectory")
#                     plt.grid()
# #                     leg = plt.legend(loc=1)
# #                     leg.get_frame().set_alpha(1)
#                     # 周波数信号(元)
#                     plt.subplot(222)
#                     plt.plot(freq, np.abs(FS), label='|F(k)|')
#                     plt.xlabel('Frequency', fontsize=20)
#                     plt.ylabel('Amplitude', fontsize=20)
#                     plt.title("Original Frequency")
#                     plt.grid()
# #                     leg = plt.legend(loc=1)
# #                     leg.get_frame().set_alpha(1)
#                     # 時間信号(処理後)
#                     plt.subplot(223)
#                     plt.plot(time-delay, HandlingData.data[s][t][key], label='f2(n)')
#                     plt.xlabel("Time", fontsize=20)
#                     plt.ylabel("Signal", fontsize=20)
#                     plt.title("Filtered Trajectory")
#                     plt.grid()
# #                     leg = plt.legend(loc=1)
# #                     leg.get_frame().set_alpha(1)
#                     # 周波数信号(処理後)
#                     plt.subplot(224)
#                     plt.plot(freq, np.abs(F2S), label='|F2(k)|')
#                     plt.xlabel('Frequency', fontsize=20)
#                     plt.ylabel('Amplitude', fontsize=20)
#                     plt.title("Filtered Frequency")
#                     plt.grid()
#                     leg = plt.legend(loc=1)
#                     leg.get_frame().set_alpha(1)
#                     plt.show()

def LabelMaker(handlingdata):
    data = np.array(handlingdata.data)
    print('----------------------------------------')
    print('| Making labels...                     |')
    print('|---------------------------------------')
    print(('| Number of labels: %s        |' % data.shape[0]))
    print('----------------------------------------')
    LABELS = []
    for i in range(data.shape[0]):
        label = np.zeros((data.shape[0]))
        label[int(i)] = 1
        Labels = []
        for j in range(data.shape[1]):
            for key, val in handlingdata.RANGE.items():
                KEY = key # Any key is ok
            labels = np.tile(label,(np.array(data[i][j][KEY]).shape[0],1))
            Labels.append(labels)
        LABELS.append(Labels)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            handlingdata.data[i][j]['LABEL'] = LABELS[i][j]
            Range_List = list(handlingdata.RANGE.values()) # OrderdDict() does not have indexing functions
            handlingdata.RANGE['LABEL'] = list(range(Range_List[0][-1], Range_List[0][-1]+data.shape[0]))
    print('----Made Labels----')
    print(np.array(LABELS).shape)
    print(handlingdata.RANGE.keys())
    print(np.array(handlingdata.data).shape)
    print('--------------------------')

def ScalingHandlingData(HandlingData, mode='DataLimit', scaling_method="normalization", isShowConsole=True):
    if isShowConsole:
        print('------------------------------')
        print('| Scaling...                    |')
        print('|-----------------------------')
        print(('| Mode: %s                   |' % mode))
        print('------------------------------')
    if mode == 'DataLimit':
        scaling_params = {}
        #HandlingData.RANGE.items():odict_items([('Torque', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15])])
        for key, val in HandlingData.RANGE.items():
            #print(key) #Torque
            # For normalization
            max_object = []
            min_object = []
            # if normalization for x, y and z is different
            # For making for sentence faster, append function is called in advance
            if key == "TACTILE_differentscale":
                max_object_xy = []
                min_object_xy = []
                max_object_z = []
                min_object_z = []
                max_object_xy_append = max_object_xy.append
                min_object_xy_append = min_object_xy.append
                max_object_z_append = max_object_z.append
                min_object_z_append = min_object_z.append
            # For standardization
            sum_object = []
            num_data_point = 0
            sum_data = 0
            for s in range(len(HandlingData.data)): #8
                # For normalization
                max_csv = []
                min_csv = []
                # If normalization for x, y and z are different
                # For making sentence faster, append function is called in advance
                if key == "TACTILE_differentscale":
                    max_csv_xy = []
                    min_csv_xy = []
                    max_csv_z = []
                    min_csv_z = []
                    max_csv_xy_append = max_csv_xy.append
                    min_csv_xy_append = min_csv_xy.append
                    max_csv_z_append = max_csv_z.append
                    min_csv_z_append = min_csv_z.append
                # For standardization
                sum_csv = []
                for t in range(len(HandlingData.data[s])): #10
                    allSizeData = np.array(HandlingData.data[s][t][key])
                    #print(allSizeData.shape) [345, 16]

                    # If normalization for x, y and z are different
                    # For making sentence faster, append function is called in advance
                    if key == "TACTILE_differentscale":
                        allSizeData_xy = []
                        allSizeData_z =  []
                        allSizeData_xy_append = allSizeData_xy.append
                        allSizeData_z_append = allSizeData_z .append
                        for i in range(len(allSizeData )):
                            sizeData_xy = []
                            sizeData_z =  []
                            sizeData_xy_append = sizeData_xy.append
                            sizeData_z_append =  sizeData_z.append
                            for j in range(len(allSizeData[i])):
                                if j % 3 == 2:
                                    sizeData_z_append(allSizeData[i][j])
                                else:
                                    sizeData_xy_append(allSizeData[i][j])
                            allSizeData_z_append(sizeData_z)
                            allSizeData_xy_append(sizeData_xy)
                    # For normalization
                    max_csv.append(np.max(allSizeData))
                    min_csv.append(np.min(allSizeData))
                    # print(max_csv, min_csv)
                # If normalization for x, y and z are different
                # For making sentence faster, append function is called in advance
                    if key == "TACTILE_differentscale":
                        max_csv_xy_append(np.max(allSizeData_xy))
                        min_csv_xy_append(np.min(allSizeData_xy))
                        max_csv_z_append(np.max(allSizeData_z))
                        min_csv_z_append(np.min(allSizeData_z))
                    # For standardization
                    num_data_point = num_data_point + allSizeData.shape[0] * allSizeData.shape[1]
                    sum_data = sum_data + np.sum(allSizeData)
                    sum_csv.append(allSizeData)
                # For normalization
                max_object.append(max_csv)
                min_object.append(min_csv)
                # If normalization for x, y and z are different
                # For making sentence faster, append function is called in advance
                if key == "TACTILE_differentscale":
                    max_object_xy_append(max_csv_xy)
                    min_object_xy_append(min_csv_xy)
                    max_object_z_append(max_csv_z)
                    min_object_z_append(min_csv_z)
                # For standardization
                sum_object.append(sum_csv)
            # For normalization
            scaling_params[key + '_input_max'] = np.max(max_object)
            scaling_params[key + '_input_min'] = np.min(min_object)
            scaling_params[key + '_value_center'] = (scaling_params[key + '_input_max'] + scaling_params[key + '_input_min']) / 2
            scaling_params[key + '_value_range'] = scaling_params[key + '_input_max'] - scaling_params[key + '_input_min']
            if key == "TACTILE_differentscale":
                scaling_params[key + '_input_max_xy'] = np.max(max_object_xy)
                scaling_params[key + '_input_min_xy'] = np.min(min_object_xy)
                scaling_params[key + '_value_center_xy'] = (scaling_params[key + '_input_max_xy'] + scaling_params[key + '_input_min_xy']) / 2
                scaling_params[key + '_value_range_xy'] = scaling_params[key + '_input_max_xy'] - scaling_params[key + '_input_min_xy']
                scaling_params[key + '_input_max_z'] = np.max(max_object_z)
                scaling_params[key + '_input_min_z'] = np.min(min_object_z)
                scaling_params[key + '_value_center_z'] = (scaling_params[key + '_input_max_z'] + scaling_params[key + '_input_min_z']) / 2
                scaling_params[key + '_value_range_z'] = scaling_params[key + '_input_max_z'] - scaling_params[key + '_input_min_z']
            # For standardization
            # Due to the difference of timesteps in each CSV, numpy.mean and numpy.std are not used.
            scaling_params[key + '_value_mean'] = sum_data / num_data_point
            squared_variance = np.sum([np.sum(np.square(np.array(timesteps) - scaling_params[key + '_value_mean']))
                                     for csvs in sum_object for timesteps in csvs ]) / num_data_point
            scaling_params[key + '_value_variance'] = np.sqrt(squared_variance)
        # Scaling Data
        for key, val in HandlingData.RANGE.items():
            if key == "Label":
                None
            else:
                for s in range(len(HandlingData.data)):
                    for t in range(len(HandlingData.data[s])):
                        if scaling_method == "normalization": # Multiplied 2 for making a scale -1 ~ 1
                            if key == "TACTILE_differentscale":  # For adjusting the scaling of contact states
    #                             HandlingData.data[s][t][key] = (HandlingData.data[s][t][key] - scaling_params[key + '_value_center']) / scaling_params[key + '_value_range'] *2 * 0.8
                                for i in range(len(HandlingData.data[s][t][key])):
                                    for j in range(len(HandlingData.data[s][t][key][i])):
                                        if j % 3 == 2:
                                            HandlingData.data[s][t][key][i][j] = (HandlingData.data[s][t][key][i][j] - scaling_params[key + '_value_center_z']) / scaling_params[key + '_value_range_z'] *2 * 0.8
                                        else:
                                            HandlingData.data[s][t][key][i][j] = (HandlingData.data[s][t][key][i][j] - scaling_params[key + '_value_center_xy']) / scaling_params[key + '_value_range_xy'] *2 * 0.8
                            else:
                                HandlingData.data[s][t][key] = (HandlingData.data[s][t][key]- scaling_params[key + '_value_center']) / scaling_params[key + '_value_range'] *2 * 0.8
                        elif scaling_method == "standardization":
                            HandlingData.data[s][t][key] = ((HandlingData.data[s][t][key] -  scaling_params[key + '_value_mean']) / scaling_params[key + '_value_variance'])
        # scaling_params = [scaling_params]
    else:
        print('Scaling Mode Error')
        print('Scaling is not done!!!')
    # print(scaling_params)
    # for i in range(8):
    #     for j in range(10):
    #         print("max",np.max(np.array(HandlingData.data[i][j]["TACTILE"])))
    #         print("min",np.min(np.array(HandlingData.data[i][j]["TACTILE"])))
    # print(HandlingData.data[0][0]["TACTILE"][0:100,0])
    return scaling_params

def GeneratingScalingParam(scaling_params, train_params):
    # Record in JSON file
    print('------------------------------')
    print('| Writing scaling parameters in JSON file...           |')
    print('------------------------------')
    print(train_params['snap_dir'] + 'scaling_params.json')
    with open(train_params['snap_dir'] + 'scaling_params.json', 'w') as json_file:
        json.dump(scaling_params, json_file)

# Since number of rows in each CSV is usually different, data size anyway should be formatted by this function
def DownSamplingData(HandlingData, num_timestep, timeInterval=10):
    print('------------------------------')
    print('| Down Sampling...           |')
    print('------------------------------')
#     print(('| TimeInterval: %3d          |' % timeInterval))
    print('| TimeInterval: Not implemented          |' )
    print('| Number of Time Steps: %3d          |' % num_timestep)
    print('------------------------------')
#     for s in range(len(HandlingData.data)):
#         for t in range(len(HandlingData.data[s])):
#             for i, key in enumerate(HandlingData.RANGE):
#                 l = len(HandlingData.data[s][t][key])
#                 HandlingData.data[s][t][key] = HandlingData.data[s][t][key][list(range(0, l, timeInterval))]
    IndexNum = num_timestep
    Index_list = []
    for s in range(len(HandlingData.data)):
        for t in range(len(HandlingData.data[s])):
            for i, key in enumerate(HandlingData.RANGE):
                if i == 0:
                    Index_list.append(len(HandlingData.data[s][t][key]))
                    # print(len(HandlingData.data[s][t][key]))
    #print(min(Index_list)) #345
    # import ipdb; ipdb.set_trace()

    for s in range(len(HandlingData.data)):
        for t in range(len(HandlingData.data[s])):
            for i, key in enumerate(HandlingData.RANGE):
                l = len(HandlingData.data[s][t][key])
                # import ipdb; ipdb.set_trace()
            # print(l,IndexNum)
            Index = random.sample(range(l),IndexNum)
            Index = np.sort(Index) # Get indices for the desired number of timesteps
            for i, key in enumerate(HandlingData.RANGE):
                A = []
                for j in Index:
                    A.append(HandlingData.data[s][t][key][j])
                HandlingData.data[s][t][key] = A
    print(np.array(HandlingData.data).shape)


def DataFormatbyStep(handlingData, nstep, changing_nextstep, inputType, outputType):
    # Formating Data for Training(Testing)
    # Take out data from dictionary
    HD = []
    for s in range(len(handlingData.data)):
        hd = []
        for t in range(len(handlingData.data[s])):
            flag = 0
            for key, val in handlingData.RANGE.items():
                if flag == 0:
                    DATA = handlingData.data[s][t][key]
                    flag = 1
                else:
                    DATA = np.c_[DATA, handlingData.data[s][t][key]]
            hd.append(np.array(DATA))
        HD.append(np.array(hd))
    handlingData.data = HD

    # Reshape training data as [CSVs, TimeSteps, Inputs]
    x_t = []
    x_tpn = []
    diff_list_sum_all2 = []
    for s in range(len(handlingData.data)):
        diff_list_sum_all = []
        for i in range(handlingData.data[s].shape[0]):
            if changing_nextstep == 1:
                ## changing nextstep ##
                print("Changing nextstep.....")
                diff_list = []
                diff_list_sum = []
                flag_dif = 0
                for k in range(len(handlingData.data[s][i])):
                    fintips = handlingData.data[s][i][k][144:216]
                    fintips = np.concatenate([fintips, handlingData.data[s][i][k][360:432]],axis=0)
                    fintips = np.concatenate([fintips, handlingData.data[s][i][k][576:648]],axis=0)
                    fintips = np.concatenate([fintips, handlingData.data[s][i][k][744:816]],axis=0)
                    if k != 0:
                        diff = fintips - old_fintips
                        # import ipdb; ipdb.set_trace()
                        diff_list.append(diff)
                        diff_list_sum.append(sum(abs(diff_list[-1])))
                        if ((sum(abs(diff_list[-1])) >= 0.07) and (flag_dif == 0)):
                            print("Index = ",k)
                            cont_index = k
                            flag_dif = 1
                    old_fintips = fintips
                diff_list_sum_all.append(diff_list_sum)
                print("Changing nextstep.....  Finish")
                ## changin nextstep Fin ##
            # import ipdb; ipdb.set_trace()
            # x_tRange = list(range(0, handlingData.data[s][i].shape[0] - nstep)) # x(t) data
            x_tRange = list(range(0, handlingData.data[s].shape[1] - nstep)) # x(t) data
            # import ipdb; ipdb.set_trace()
            x_t.append(handlingData.data[s][i, x_tRange])
            # x_t.append(handlingData.data[s][i][x_tRange])

            if changing_nextstep == 1:
                x_tpnRange = list(range(1,cont_index))
                x_tpnRange = np.concatenate([x_tpnRange, list(range(cont_index+nstep-1, handlingData.data[s].shape[1]))],axis=0)   # x(t+nstep) data
                print(x_tpnRange)
            else:
                # x_tpnRange = list(range(nstep, handlingData.data[s].shape[0]))   # x(t+nstep) data
                x_tpnRange = list(range(nstep, handlingData.data[s][i].shape[0]))   # x(t+nstep) data
            x_tpn.append(handlingData.data[s][i, x_tpnRange])
            # x_tpn.append(handlingData.data[s][i][x_tpnRange])
            # diff_all = []
            # diff_all3 = []
            # for t in range((handlingData.data[s][i].shape[1])):
            #     diff_all2 = []
            #     for k in range(len(handlingData.data[s][i][:,t])-1):
            #         diff = handlingData.data[s][i][:,t][k+1] - handlingData.data[s][i][:,t][k]
            #         diff_all2.append(diff)
            #     diff_all.append(diff_all2)
            # import ipdb; ipdb.set_trace()
        diff_list_sum_all2.append(diff_list_sum_all)
    # import ipdb; ipdb.set_trace()
    x_t = np.array(x_t)
    x_tpn = np.array(x_tpn)

    # Change an order of columns according to the desired input order in inputType
    X_T = []
    flag = 0
    # print(len(handlingData.RANGE["TACTILE"]), handlingData.RANGE["Label"])
    for typ in inputType:
        if flag == 0:
            X_T = x_t[:,:,handlingData.RANGE[typ]]
            flag = 1
        else:
            X_T = np.c_[X_T, x_t[:,:,handlingData.RANGE[typ]]]
        # print(handlingData.RANGE[typ])
    X_TPN = []
    flag = 0
    for typ in outputType:
        if flag == 0:
            X_TPN = x_tpn[:,:,handlingData.RANGE[typ]]
            flag = 1
        else:
            X_TPN = np.c_[X_TPN, x_tpn[:,:,handlingData.RANGE[typ]]]

    print('----ReshapedData Shape [CSV, Timestep, Input}] ----')
    print(np.array(X_T).shape)
    print(np.array(X_TPN).shape)
    print('------------------------------------------------------------------')
    return X_T, X_TPN


def DataFormatbyStep_FNN(handlingData, nstep, inputType, outputType):
    # Formating Data for Training(Testing)
    # Take out data from dictionary
    HD = []

    ###

    for s in range(len(handlingData.data)): #8
        hd = []
        for t in range(len(handlingData.data[s])): #10
            flag = 0
            for key, val in handlingData.RANGE.items(): #Torque
                #print(key)
                if flag == 0:
                    DATA = handlingData.data[s][t][key]
                    flag = 1
                else:
                    DATA = np.c_[DATA, handlingData.data[s][t][key]]
            hd.append(np.array(DATA))
        HD.append(np.array(hd))
    handlingData.data = HD
    #print(np.array(handlingData.data).shape)

    # Reshape training data as [CSVs, TimeSteps, Inputs]
    x_t = []
    x_tpn = []
    for s in range(len(handlingData.data)): #8
        for i in range(handlingData.data[s].shape[0]): #10
            x_tRange = list(range(0, handlingData.data[s].shape[1] - nstep)) # x(t) data
            x_tpnRange = list(range(nstep, handlingData.data[s].shape[1]))   # x(t+nstep) data
            x_t.append(handlingData.data[s][i, x_tRange])
            x_tpn.append(handlingData.data[s][i, x_tpnRange])
    x_t = np.array(x_t)
    x_tpn = np.array(x_tpn)

    #print(x_t.shape, x_tpn.shape)

    # Change an order of columns according to the desired input order in inputType
    X_T = []
    flag = 0
    for typ in inputType:
        if flag == 0:
            X_T = x_t[:,:,handlingData.RANGE[typ]]
            flag = 1
        else:
            X_T = np.c_[X_T, x_t[:,:,handlingData.RANGE[typ]]]
        print(handlingData.RANGE[typ])
    X_TPN = []
    flag = 0
    for typ in outputType:
        if flag == 0:
            X_TPN = x_tpn[:,:,handlingData.RANGE[typ]]
            flag = 1
        else:
            X_TPN = np.c_[X_TPN, x_tpn[:,:,handlingData.RANGE[typ]]]
    print('----ReshapedData Shape [CSV, Timestep, Input}] ----')
    print(np.array(X_T).shape)
    print(np.array(X_TPN).shape)
    print('------------------------------------------------------------------')
    return X_T, X_TPN
    

def ReshapeForNN(Learning_train, Learning_teacher):
    print(Learning_train.shape)
    for i in range(Learning_train.shape[0]):
        if i == 0:
            reshape_train = Learning_train[i]
        else:
            reshape_train = np.vstack([reshape_train, Learning_train[i]])
    for i in range(Learning_teacher.shape[0]):
        if i == 0:
            reshape_teacher = Learning_teacher[i]
        else: reshape_teacher = np.vstack([reshape_teacher, Learning_teacher[i]])
    print('----ReshapedData Shape----')
    print(np.array(reshape_train).shape)
    print(np.array(reshape_teacher).shape)
    print('------------------------------------------------------------------')
    return reshape_train, reshape_teacher

def ReshapeTimeSeriesData(x_t, timeWindow):
    #Time Series Inputs
    print('-----------------------------------------------')
    print('| Converting to Time Series Inputs...         |')
    print('-----------------------------------------------')
    print('| Time Window: %3d                            |' % timeWindow)
    print('-----------------------------------------------')
    x_t_SERIES_CSV = []
    for i in range(len(x_t)):
        x_t_SERIES = []
        for j in range(len(x_t[i])-timeWindow):
            timesteps = list(range(j, j+timeWindow))
            x_t_series = x_t[i][timesteps,:]
            x_t_SERIES.append(x_t_series)
        x_t_SERIES_CSV.append(x_t_SERIES)
    x_t_SERIES_CSV = np.array(x_t_SERIES_CSV)
    print('----ReshapedData Shape----')
    print(np.array(x_t_SERIES_CSV).shape)
    print('------------------------------------------------------------------')
    return x_t_SERIES_CSV

def Euclidean(handlingData, Learning_train, Learning_teacher, thresholdDist, calcBaseType=['JOINT','TACTILE']):
    print('------------------------------')
    print('| Euclidean...               |')
    print('------------------------------')
    print(('| Threshold: %.2f            |' % thresholdDist))
    print('------------------------------')
    distance = []
    # ユークリッド距離の計算基準とするステートを作る
    # (デフォルト引数では,モータ・タクタイルを計算に使う)
    lb_x_t = []
    x_tRange= {}
    tmphead = 0
    for typ in calcBaseType:
        lb_x_t += handlingData.RANGE[typ]
        x_tRange[typ] = np.array(list(range(len(handlingData.RANGE[typ])))) + tmphead
        tmphead = x_tRange[typ][-1] + 1
    print(lb_x_t)
    state = Learning_train[:, lb_x_t]
    print(Learning_train.shape)
    state_idx = np.array(list(range(0,state.shape[0]))) # 抽出するインデックスを保持する変数(はじめはすべてのインデックスが候補)
    euc_idx = state_idx
    tf_idx = np.tile(True, state.shape[0])
    i = 0
    # すべてのステートに対するユークリッド距離を見る
    # (ただし、計算量削減のため、ある基準ステートに対して
    #  ユークリッド距離が閾値を下回ったステートは、以後基準ステートとして取らない)
    while(i < len(euc_idx)):
        # 距離計算基準となるステートを抽出
        idx = euc_idx[i]
        instance = state[idx]
        # 抽出したステートと他すべてのステートとのユークリッド距離を計算
        distance = np.sqrt(np.sum((state - instance)**2, axis=1))
        # 閾値より小さいユークリッド距離を持つステートをFalseとした行列
        idx = ~(distance < thresholdDist)
        idx[i] = True   # 自分自身との距離は必ず0になってしまいFalse判定されてしまうのを避けるためにTrueにする
        # 論理積を取り,閾値以下の要素インデックスをFalseにする(Trueの要素がEuclideanでの抽出対象)
        tf_idx = tf_idx & idx
        # 計算対象となるステートのインデックスを更新
        euc_idx = state_idx[tf_idx]
        # 計算基準のインデックスのインクリメント
        i = i + 1
    print('Euclidean result: %d/%d' % (len(euc_idx), len(state_idx)))
    return Learning_train[euc_idx], Learning_teacher[euc_idx]

def PruneUnstableHandlingState(handlingData, Learning_train, Learning_teacher, unstableForceMagnitude):
    print('------------------------------')
    print('| Force Pruning...           |')
    print('------------------------------')
    print('| Threshold: %4d            |' % unstableForceMagnitude)
    print('------------------------------')
    sixaxisData = Learning_train[:,handlingData.RANGE['FORCE']]
    # もともとfor文だったが,遅い上,必要性が不明だったためdeepcopyに変更した(2017年1月4日小笠)
    # for i in xrange(sixaxisData.shape[0]):
    #     if i == 0:
    #         sixaxis = sixaxisData[i]
    #     else:
    #         sixaxis = np.vstack([sixaxis, sixaxisData[i]])
    sixaxis = copy.deepcopy(sixaxisData)
    # 小笠変更ここまで
        # -1~+1にスケーリングされたデータを元の単位系に戻す
    LIMIT = LIMIT_SIXAXIS
    # ReScaling Data
    sixaxis = sixaxis * ((LIMIT[:, 1] - LIMIT[:, 0]) / 2.) + ((LIMIT[:, 1] + LIMIT[:, 0]) / 2.)
    # 4指に対応した(2017年1月4日小笠)
    # forceMagnitudeThumb = np.sqrt(sixaxis[:,0] ** 2 + sixaxis[:,1] ** 2 + sixaxis[:,2] ** 2)
    # forceMagnitudeIndex = np.sqrt(sixaxis[:,6] ** 2 + sixaxis[:,7] ** 2 + sixaxis[:,8] ** 2)
    # forceMagnitude = (forceMagnitudeThumb + forceMagnitudeIndex) / 2
    fin_num_total = 4  # 指の数
    forceMagnitudeList = []
    forceMagnitude = 0 # すべての指の力の大きさを足し合わせる
    for fin_num in range(fin_num_total):
        # 後々listの中身をそれぞれで使う可能性を見据えて冗長な書き方をしている
        forceMagnitudeList.append(np.sqrt(
            sixaxis[:,fin_num * 6] ** 2 +
            sixaxis[:,fin_num * 6 + 1] ** 2 +
            sixaxis[:,fin_num * 6 + 2] ** 2
        ))
        forceMagnitude = forceMagnitude + forceMagnitudeList[fin_num]
    forceMagnitude = forceMagnitude / fin_num_total
    # 小笠変更ここまで
#     print(forceMagnitude)
#     sns.distplot(forceMagnitude)
#     plt.show()
    # 合力の大きさが閾値を越えているインデックスを取得する
    idx = forceMagnitude > unstableForceMagnitude
    pruned_train = Learning_train[idx]
    # teacherは既にインデックスが1ずれているので,
    # trainのインデックスと同じところを指定すればtとt+1の対応がとれる
    pruned_teacher = Learning_teacher[idx]
    # 枝刈り結果表示用計算
    train_all = len(Learning_train)
    pruned_sum = len(pruned_train)
    print("Pruned [%d/%d]" % (pruned_sum, train_all))
    pruned_train = np.array(pruned_train)
    pruned_teacher = np.array(pruned_teacher)
    # for debug
#     for i in xrange(pruned_train.shape[0]):
#         print "[%d] train:%d, teacher:%d" % (i, len(pruned_train[i]), len(pruned_teacher[i]))
    return pruned_train, pruned_teacher

def RandomSampling(Learning_train,  Learning_teacher, samplingNum):
    print('------------------------------')
    print('| Random Sampling...         |')
    print('------------------------------')
    L = len(Learning_train)
    if len(Learning_train) <= samplingNum:
        samplingNum = len(Learning_train)
        print('over number of raw data RandomSampling disregarded')
        return Learning_train, Learning_teacher
    else:
        np.random.seed(int(time.time()))
        randomnumber = int(time.time())
        index = np.random.randint(0,len(Learning_train), samplingNum)
        index = np.sort(index)
        # print(index)
        Learning_train = Learning_train[index]
        Learning_teacher = Learning_teacher[index]
        print("Sampled [%d/%d]" % (len(index), L))
        # print(randomnumber)
        print(Learning_train.shape, Learning_teacher.shape)
        return Learning_train, Learning_teacher, randomnumber

def Whitening_ZCA(inputs, fudge=1E-18):
    sigma = np.dot(inputs, inputs.T)/inputs.shape[1] #Correlation matrix
    U,S,V = np.linalg.svd(sigma) #Singular Value Decomposition
    epsilon = 0.1                #Whitening constant, it prevents division by zero
    ZCAMatrix = np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T)                     #ZCA Whitening matrix
    return np.dot(ZCAMatrix, inputs)   #Data whitening

def DeleteTimeColumn(HandlingData):
    print('-----------------------------------')
    print('| Delete Time Column...           |')
    print('-----------------------------------')
    print(np.array(HandlingData).shape)
    l = HandlingData.shape[2]
    HandlingData = HandlingData[:,:, list(range(1, l))]
    print(np.array(HandlingData).shape)
    return HandlingData

def TrainTestDivider(inputSignal, teacherSignal,  testSize):
    print('--------------------------------------------------------------')
    print('| Make training and test data ...                   |')
    print('--------------------------------------------------------------')
    #Make learning & testing data (input, teacher)
    inputSignal, inputSignal_test, teacherSignal, teacherSignal_test = train_test_split(inputSignal, teacherSignal, test_size=testSize, random_state=int(time.time()))
    print(inputSignal.shape, inputSignal_test.shape)
    print(teacherSignal.shape, teacherSignal_test.shape)

    return inputSignal, inputSignal_test, teacherSignal, teacherSignal_test