import glob
import math
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def saveCoordinatesImage(handlingData, tactileCoordinatesName):
    tactileCoordinates = handlingData.data[0][0]["Coordinates_Tactile"]
    # fig = plt.figure()
    # ax = Axes3D(fig)
    # ax.plot(tactileCoordinates[0,0::3],tactileCoordinates[0,1::3],
    # tactileCoordinates[0,2::3], color="gray",marker="o", linestyle='None')
    # plt.savefig("{}.jpg".format(tactileCoordinatesName[count*3]))

    count = 0
    marker_list = ["${}$".format(i) for i in range(16)]

    for i in range(384):
        # import ipdb; ipdb.set_trace()
        fig = plt.figure()
        ax = Axes3D(fig)
        # ax.set_title("{} {}".format(i, tactileCoordinatesName[i*3]),size=20)
        ax.plot(tactileCoordinates[0,0::3],tactileCoordinates[0,1::3],
        tactileCoordinates[0,2::3], color="gray",marker="o", linestyle='None')
        ax.plot(tactileCoordinates[0,i*3],tactileCoordinates[0,i*3+1],
        tactileCoordinates[0,i*3+2],color="red", marker="o", linestyle='None')
        # for j in range(i):
        #     ax.plot(tactileCoordinates[0,count*3:(count+i)*3:3],tactileCoordinates[0,count*3+1:(count+i)*3+1:3],
        #     tactileCoordinates[0,count*3+2:(count+i)*3+2:3],color="black", marker=marker_list[j], linestyle='None')
        ax.set_title("{} {}".format(i, tactileCoordinatesName[i*3]))
        plt.savefig("image_list/{}_{}.jpg".format(i,tactileCoordinatesName[i*3]))
    # for f in [[16,16,16,16,8],[16,16,16,16,8],[16,16,16,16,8],[16,16,16,8],[16]*7]:
    #     for i in f:
    #         # import ipdb; ipdb.set_trace()
    #         fig = plt.figure()
    #         ax = Axes3D(fig)
    #         ax.plot(tactileCoordinates[0,0::3],tactileCoordinates[0,1::3],
    #         tactileCoordinates[0,2::3], color="gray",marker="o", linestyle='None')
    #         ax.plot(tactileCoordinates[0,count*3:(count+i)*3:3],tactileCoordinates[0,count*3+1:(count+i)*3+1:3],
    #         tactileCoordinates[0,count*3+2:(count+i)*3+2:3],color="red", marker=marker_list[0:i], linestyle='None')
    #         # for j in range(i):
    #         #     ax.plot(tactileCoordinates[0,count*3:(count+i)*3:3],tactileCoordinates[0,count*3+1:(count+i)*3+1:3],
    #         #     tactileCoordinates[0,count*3+2:(count+i)*3+2:3],color="black", marker=marker_list[j], linestyle='None')
    #         plt.savefig("image_list/{}.jpg".format(tactileCoordinatesName[count*3]))
    #         count +=i

def saveCoordinatesImage_(loadDirs, tactileCoordinatesName):
    # import ipdb; ipdb.set_trace()
    files = glob.glob(loadDirs[0]+"*.csv")
    data = pd.read_csv(files[0])
    i=12
    tactileCoordinates = data.values[:,1246+i:2398+i]
    import ipdb; ipdb.set_trace()

    count = 0
    for f in [[16,16,16,16,8],[16,16,16,16,8],[16,16,16,16,8],[16,16,16,8],[16]*7]:
        for i in f:
            # import ipdb; ipdb.set_trace()
            fig = plt.figure()
            ax = Axes3D(fig)
            ax.plot(tactileCoordinates[0,0::3],tactileCoordinates[0,1::3],
            tactileCoordinates[0,2::3], color="gray",marker="o", linestyle='None')
            ax.plot(tactileCoordinates[0,count*3:(count+i)*3:3],tactileCoordinates[0,count*3+1:(count+i)*3+1:3],
            tactileCoordinates[0,count*3+2:(count+i)*3+2:3],color="red", marker="o", linestyle='None')
            plt.savefig("image/{}.jpg".format(tactileCoordinatesName[count*3]))
            count +=i

def saveCoordinatesImage_j(loadDirs, jointCoordinatesName):
    # import ipdb; ipdb.set_trace()
    files = glob.glob(loadDirs[0]+"*.csv")
    data = pd.read_csv(files[0])
    i=0
    jointCoordinates = data.values[:,1198+i:1246+i]
    import ipdb; ipdb.set_trace()

    count = 0
    for i in [4,4,4,4]:
        # import ipdb; ipdb.set_trace()
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.plot(jointCoordinates[0,0::3],jointCoordinates[0,1::3],
        jointCoordinates[0,2::3], color="gray",marker="o", linestyle='None')
        ax.plot(jointCoordinates[0,count*3:(count+i)*3:3],jointCoordinates[0,count*3+1:(count+i)*3+1:3],
        jointCoordinates[0,count*3+2:(count+i)*3+2:3],color="red", marker="o", linestyle='None')
        plt.savefig("image/{}.jpg".format(jointCoordinatesName[count*3]))
        count +=i