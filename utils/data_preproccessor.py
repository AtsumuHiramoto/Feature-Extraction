#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
# import ObjectSIzeShapeAnalyzer as osa
# import SensorCoordinatesManager as spm

class DataPreprocessor(object):
    def __init__(self) -> None:
        self.handling_data = []
    def load_handling_dataset(self, load_dir):
        """
        Function to load dataset
        Parameters
        ----------
        load_dir: str
            Directory name to load which contains csv. 
            You can use regular expression.
            e.g. "hoge/*/*/"
        """
        self.load_csv_file_list = glob.glob(load_dir + "*.csv")
        if len(self.load_csv_file_list)==0:
            print("{} doesn't have csv file".format(load_dir))
            exit()
        for load_csv_file in self.load_csv_file_list:
            load_csv = pd.read_csv(load_csv_file)
            import ipdb; ipdb.set_trace()
            self.handling_data.append(load_csv)
        return self.handling_data
    
    def make_cache_data():
        pass
    
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
