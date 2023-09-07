#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import fnmatch
import csv
import numpy as np
import pandas as pd
import glob
from PIL import Image
import re
import matplotlib.pyplot as plt
import DataPreProcessor as dpp
# Limit of Sensor Data
# モータ角LIMIT
LIMIT_MOTOR = np.array([[-28.80, 27.66], [-12.19, 94.20], [-10.47, 98.00], [-13.80, 93.55],
                        [-28.80, 27.66], [-12.19, 94.20], [-10.47, 98.00], [-13.80, 93.55],
                        [-28.80, 27.66], [-12.19, 94.20], [-10.47, 98.00], [-13.80, 93.55],
                        [ 25.63,  80.94], [-7.03, 61.10], [-11.62, 94.69], [ -9.90, 99.28]]) # , [0, 9500]])
LIMIT_DMOTOR = np.array([[-28.80, 27.66], [-12.19, 94.20], [-10.47, 98.00], [-13.80, 93.55],
                        [-28.80, 27.66], [-12.19, 94.20], [-10.47, 98.00], [-13.80, 93.55],
                        [-28.80, 27.66], [-12.19, 94.20], [-10.47, 98.00], [-13.80, 93.55],
                        [ 25.63,  80.94], [-7.03, 61.10], [-11.62, 94.69], [ -9.90, 99.28]]) # , [0, 9500]])
# 6軸力覚センサLIMIT
# 拇指，示指共通
LIMIT_SIXAXIS = np.array([-500.00, 500.00])
# 物体サイズLIMIT
LIMIT_SIZE = np.array([0.01, 0.15])
# 物体形状情報LIMIT
LIMIT_SHAPE = np.array([-2, 2])
# LIMIT_SHAPE = np.array([[-1, 1]]) # Curvature
# 画像センサLIMIT
LIMIT_IMAGE = np.array([0, 255.0])
IMAGE_SIZE = 96
# 触覚センサLIMIT
LIMIT_TACTILE = np.array([0])
# Threshold
DROP_FORCE = 4  # 6軸合力がこの値を下回ったら対象物落下とみなす閾値
#===============================================================================
# Methods 
#===============================================================================
def LoadCSV(loadFile, fileCount=0):
    ''' Load External CSV File
    1行目：ラベル，2行目以降：数値 となっているcsvファイルを読み込む
    :type loadFile: string
    :param loadFile: 読み込むcsvファイルの場所を示す文字列 '''
    [dataDir, dataFile] = os.path.split(loadFile)
    # Check
    if not os.path.isfile(loadFile):
        print('File not found.')
        return
#     print('  [%d] Load [%s]...' % (fileCount,dataFile))
    df = pd.read_csv(loadFile)
    return df

def SaveCSV(saveFileName, label, data):
    csvfile = csv.writer(file(saveFileName, 'w'))
    csvfile.writerow(label)
    for row in data:
        csvfile.writerow(row)

def LoadImage(loadDirs):
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
            img = nomalization(np.array(img).astype(np.float32),[0,255],[0,255])
            data.append(img)
#         print('IMAGE DATA SHAPE1') 
#         print(np.array(data).shape)
        DATA.append(np.array(data))
    print('IMAGE DATA SHAPE2')
    print(np.array(DATA).shape)
    return DATA

def SaveImage(self, file_name, image_ndarray, cols=8):
    # 画像数, 幅, 高さ
    count, w, h = image_ndarray.shape
    # 縦に画像を配置する数
    rows = int((count - 1) / cols) + 1
    # 復数の画像を大きな画像に配置し直す
    canvas = Image.new("RGB", (w * cols + (cols - 1), h * rows + (rows - 1)), (0x80, 0x80, 0x80))
    for i, image in enumerate(image_ndarray):
        # 横の配置座標
        x_i = int(i % cols)
        x = int(x_i * w + x_i * 1)
        # 縦の配置座標
        y_i = int(i / cols)
        y = int(y_i * h + y_i * 1)
        out_image = Image.fromarray(np.uint8(image))
        canvas.paste(out_image, (x, y))
    if os.path.isdir('./imagesfromlayer/'):
        canvas.save('./imagesfromlayer/' + file_name.replace('./',''), "PNG")
    else:
        os.mkdir('./imagesfromlayer/')
        canvas.save('./imagesfromlayer/' + file_name.replace('./',''), "PNG")

def LoadFile(loadFilePath):
    """
    files = os.listdir(loadFilePath)
    dic = {}
    print(('Loading [%s/]' % loadFilePath))
    for i,file in enumerate(files):
        if os.path.splitext(loadFilePath + '/' + file)[1] == '.csv':
            df = LoadCSV(loadFilePath + '/' + file, i)
            df = df.T.dropna().T     # 欠損値を削除
            header = df.columns.values
            dic[file] = df
        else:
            print(file+' is not a csv file. Skipping.')
    return pd.Panel(dic), header
    """
    files = os.listdir(loadFilePath)
    dic = {}
    FN=[]
    print(('Loading [%s/]' % loadFilePath))
    i = 0
    for file in files:
        if os.path.splitext(loadFilePath + '/' + file)[1] == '.csv':
            df = LoadCSV(loadFilePath + '/' + file, i)
            df = df.T.dropna().T     # 欠損値を削除
            header = df.columns.values
            dic[i] = df
            FN.append(file)
            #print("df", i, df.values)
            i+=1
        else:
            print(file+' is not a csv file. Skipping.')
    return dic, header, FN

def GetDirectoryList(loadDir):
    items = os.listdir(loadDir)
    dirs = []
    for item in items:
        if os.path.isdir(os.path.join(loadDir, item)):
            dirs.append(item)
    dirs.sort()
    return dirs

def GetCSVFileList(loadDir):
    files = os.listdir(loadDir)
    csvfiles = fnmatch.filter(files, '*.csv')
    csvfiles.sort()
    return csvfiles

class CHandlingData(object):
    def __init__(self, data=[], Objectlabel=[], fileNames=[], RANGE=None):
        self.data = data
        self.LIMIT = {"JOINT":LIMIT_MOTOR, \
              "FORCE":LIMIT_SIXAXIS, \
              "DMOTOR" : LIMIT_DMOTOR, \
              "IMAGE" : LIMIT_IMAGE, \
              "TACTILE" : LIMIT_TACTILE, \
              "SIZE":LIMIT_SIZE, \
              "SHAPE":LIMIT_SHAPE}
        self.RANGE = RANGE
        self.fileNames = fileNames
        self.Objectlabel = Objectlabel

if __name__ == '__main__':
    loadDirs = "/home/funabashi/workspace/AnalysisDataforAllegroHand/test/2finsph60/csv*"
    loadDirs = glob.glob(loadDirs)
    inputType = ['TIME','JOINT','FORCE']
    outputType = ['MOTOR']
    HD = []
    FN = []

    Objectlabel = []
    for loadDir in loadDirs:
        loadDir_sep = loadDir.split('/')
        Objectlabel.append(loadDir_sep[-2])
        # CSVファイルの読み込み
        handlingData, INPUT = LoadFile(loadDir)
        files = handlingData.axes[0]
        for i, file in enumerate(files):
            print('  [%d]:%s' % (i, file))
        FN.append(list(map(str,files)))
#         print(INPUT)
#         print(handlingData.shape)
#         print(handlingData)
        new_handlingData = []
        dic = {}
        for f, key in enumerate(handlingData):
            INPUT_DATA = {}
            ALL_RANGE = {}
            index = 0
            last_index = 0
            for i in range(len(inputType)):
                flag = 0
                for j in range(len(INPUT)):
                    # 文字列の先頭でパターンがマッチするかどうかを判定
                    if re.match(inputType[i], INPUT[j], re.IGNORECASE):
                        data = handlingData.values[f][:,j]
                        index +=1
                        if flag == 0:
                            DATA = data
                            flag = 1
                        else:
                            DATA = np.c_[DATA, data]
#                     print(INPUT[j])
#                     print(inputType[i])
                if flag != 0:
#                     print(DATA)
                    INPUT_DATA[inputType[i]] = DATA
                    RANGE = list(range(last_index, index))
                    ALL_RANGE[inputType[i]] = RANGE 
                last_index = index
#             plt.plot(INPUT_DATA['TIME'],INPUT_DATA['JOINT'])
#             plt.show()
            new_handlingData.append(INPUT_DATA)
        HD.append(new_handlingData)
#         objectSize = CalculateObjectSize_Simple(handlingData.values)
        # 物体サイズを行列に付加
        dic = {}
#         for i, key in enumerate(handlingData):
#             handlingData[key]['Size'] = objectSize[i]
#             dic[key] = handlingData[key]
        # Panelを作り直さないとなぜかSizeが追加されない..
        handlingData = pd.Panel(dic)
    print(ALL_RANGE.values())
    print(np.array(HD).shape)
    handlingData = CHandlingData(HD, Objectlabel, FN, ALL_RANGE)
    dpp.DownSamplingData(handlingData, timeInterval=4)
    dpp.CheckLimit(HandlingData=handlingData)
    dpp.ScalingHandlingData(HandlingData=handlingData, mode='SpecLimit')
    print(np.array(handlingData.data).shape)
    x_t, x_tpn = dpp.DataFormatbyStep(handlingData, nstep=20)
