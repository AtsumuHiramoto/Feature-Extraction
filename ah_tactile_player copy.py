import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import matplotlib
matplotlib.use('TkAgg')
# csv_file = '/home/handlingteam/allegro_hand_ros_catkin_fork_ver2/src/allegro_hand_ros_catkin_client/record_data/20230629_163721_allegro_ah.csv'
csv_file = "../dataset/cap_raw/shampool/0/20230907_163714_allegro_shampool_1_1.csv"
plot_Hz = 20

class AHTactilePlayer(object):
    def __init__(self, csv_file, plot_Hz):
        self.csv_file = csv_file
        self.plot_Hz = plot_Hz
        self.image_list = []
        self.fig = plt.figure(figsize=(10,10))
        # self.fig = plt.figure(figsize=(20,20))
        self.ax1 = self.fig.add_subplot(1,2,1)
        self.ax2 = self.fig.add_subplot(1,2,2)
        # self.ax = self.fig
        self.ax1.axis("off")
        self.ax2.axis("off")
        # plt.xlim(0,500)
        # plt.ylim(0,500)
        # plt.xlim(-500,1000)
        # plt.ylim(-500,1000)
        self.set_patch_arrangement()
        self.ah_tactile_player(csv_file=csv_file, plot_Hz=plot_Hz)

    def ah_tactile_player(self, csv_file, plot_Hz):
        # csv_file : str path -> dataframe
        # tactile_df = pd.read_csv(csv_file)
        tactile_df = csv_file
        self.start_time = tactile_df['ClientTimeStamp'][0]
        # import ipdb; ipdb.set_trace()
        count = 0
        plot_duration = 1.0 / float(plot_Hz)
        tmp_time_stamp = tactile_df["ClientTimeStamp"][0]
        for i in range(len(tactile_df)):
        # for i in range(10):
            time_stamp = tactile_df["ClientTimeStamp"][i]
            if (time_stamp - tmp_time_stamp) > plot_duration:
                self.simple_tactile_viz(tactile_df[i:i+1])
                tmp_time_stamp = time_stamp
                count = 0
            else:
                count += 1
                print(count)
            # time.sleep(0.01)

    def simple_tactile_viz(self, merge_df):
        # import ipdb; ipdb.set_trace()
        tac_df = merge_df.filter(like='Tactile',axis=1)
        self.finger_name = ['Index', 'Middle', 'Little', 'Palm']
        self.patch_name = ['IndexTip_TactileB00', 'IndexPhalange_TactileB00', 'IndexPhalange_TactileB01', 'IndexPhalange_TactileB02',
                        'MiddleTip_TactileB00', 'MiddlePhalange_TactileB00', 'MiddlePhalange_TactileB01', 'MiddlePhalange_TactileB02',
                        'LittleTip_TactileB00', 'LittlePhalange_TactileB00', 'LittlePhalange_TactileB01', 'LittlePhalange_TactileB02',
                        'ThumbTip_TactileB00', 'ThumbPhalange_TactileB00', 'ThumbPhalange_TactileB01',
                        'Palm_TactileB00', 'Palm_TactileB01', 'Palm_TactileB02']

        patch_all = True
        if patch_all == True:
            # for patch_index in range(0,2):
                # patch_df = (merge_df.filter(like=self.patch_name[patch_index],axis=1))
            self.plot_all(merge_df)
            # plt.pause(0.001)
            # self.ax.remove()

    def plot_all(self, merge_df):
        initial_position = [
            [14,25], [15,20], [15,15], [15,10], #index
            [7,25], [8,20], [8,15], [8,10], #middle
            [0,25], [1,20], [1,15], [1,10], #little
            [10,-12], [11,-5], [11,0], #thumb
            # [3.5,0], [3.5,5], [10.5,5] #palm
            [10.5,5], [3.5,5], [3.5,0] #palm
        ]
        # import ipdb; ipdb.set_trace()
        plot_time = merge_df['ClientTimeStamp'].values[-1] - self.start_time
        x_list = []
        y_list = []
        marker_size_list = []
        color_list = []
        start = time.time()
        scale = 200
        marker_size_lim = 20
        sensor_rate_x = 0.1
        sensor_rate_y = 0.1
        # sensor_rate_z = 0.05
        sensor_rate_z = 0.1
        # self.ax = self.fig.add_subplot(111)
        self.ax = self.fig.add_subplot(1,2,1)
        self.ax.axis("off")
        plt.xlim(-scale - 11.5*scale, scale*22 + 11.5*scale)
        plt.ylim(scale*(-13), scale*33)
        for patch_index in range(0,18):
            patch_df = (merge_df.filter(like=self.patch_name[patch_index],axis=1))
            import ipdb; ipdb.set_trace()
            patch_arrangement = self.patch_all[patch_index]
            if patch_index==12:
                patch_arrangement = patch_arrangement*(-1) + 5
            if patch_index==13 or patch_index==14:
                patch_arrangement = patch_arrangement*(-1) + 3
            # import ipdb; ipdb.set_trace()
            sensor_num = patch_arrangement.shape[0]
            # import ipdb; ipdb.set_trace()
            # tactile_data = np.array(patch_df).reshape([16,3])
            # print(sensor_num)
            tactile_data = np.array(patch_df).reshape([sensor_num,3])
            for s in range(sensor_num):
                marker_size = marker_size_lim + abs(int(tactile_data[s,2])*sensor_rate_z) # z axis value
                marker_color = 'g' if tactile_data[s,2] > 0  else 'b'
                if patch_index in [1,2,3,5,6,7,9,10,11,12]:
                    ts_x = int(-tactile_data[s,1]*sensor_rate_x) # x axis translation
                    ts_y = int(tactile_data[s,0]*sensor_rate_y) # y axis translation
                elif patch_index in [0,4,8,13,14,15,16,17]:
                    ts_x = int(tactile_data[s,1]*sensor_rate_x) # x axis translation
                    ts_y = int(-tactile_data[s,0]*sensor_rate_y) # y axis translation
                x_list.append((patch_arrangement[s][0]+initial_position[patch_index][0])*scale + ts_x)
                y_list.append((patch_arrangement[s][1]+initial_position[patch_index][1])*scale + ts_y)
                marker_size_list.append(marker_size)
                color_list.append(marker_color)

        print(time.time()-start)
        plt.title(str(plot_time))
        image = self.ax.scatter(x_list,y_list, s=marker_size_list, marker='o', color=color_list)
        self.ax2 = self.fig.add_subplot(1,2,2)
        image = self.ax2.scatter(x_list,y_list, s=marker_size_list, marker='o', color=color_list)
        # plt.show()
        plt.pause(0.001)
        self.ax.remove()

    def set_patch_arrangement(self):
        self.patch_all = []
        patch_0 = torch.tensor(
            [[2,1], [2,2], [2,3], [2,4], [2,5],
            [3,1], [3,2], [3,3], [3,4], [3,5],
            [0,1], [0,2], [0,3], [0,4],
            [1,1], [1,2], [1,3],
            [5,1], [5,2], [5,3], [5,4],
            [4,1], [4,2], [4,3],
            [0,0], [1,0], [2,0], [3,0], [4,0], [5,0]
            ]
        , dtype=torch.float32); self.patch_all.append(patch_0)
        patch_1 = torch.tensor(
            [[3,0], [3,1], [3,2], [3,3],
            [2,0], [2,1], [2,2], [2,3],
            [1,0], [1,1], [1,2], [1,3],
            [0,0], [0,1], [0,2], [0,3],
            ]
        , dtype=torch.float32); self.patch_all.append(patch_1)
        patch_2 = torch.tensor(
            [[0,3], [0,2], [0,1], [0,0],
            [1,3], [1,2], [1,1], [1,0],
            [2,3], [2,2], [2,1], [2,0],
            [3,3], [3,2], [3,1], [3,0],
            ]
        , dtype=torch.float32); self.patch_all.append(patch_2)
        patch_3 = patch_1.clone(); self.patch_all.append(patch_3)
        patch_4 = patch_0.clone(); self.patch_all.append(patch_4)
        patch_5 = patch_1.clone(); self.patch_all.append(patch_5)
        patch_6 = patch_2.clone(); self.patch_all.append(patch_6)
        patch_7 = patch_1.clone(); self.patch_all.append(patch_7)
        patch_8 = patch_0.clone(); self.patch_all.append(patch_8)
        patch_9 = patch_1.clone(); self.patch_all.append(patch_9)
        patch_10 = patch_2.clone(); self.patch_all.append(patch_10)
        patch_11 = patch_1.clone(); self.patch_all.append(patch_11)
        patch_12 = patch_0.clone(); self.patch_all.append(patch_12)
        patch_13 = patch_1.clone(); self.patch_all.append(patch_13)
        patch_14 = patch_2.clone(); self.patch_all.append(patch_14)
        patch_15 = torch.tensor(
            [[0,3], [1,3], [2,3], [3,3], [4,3], [5,3],
            [0,2], [1,2], [2,2], [3,2], [4,2], [5,2],
            [0,1], [1,1], [2,1], [3,1], [4,1], [5,1],
            [0,0], [1,0], [2,0], [3,0], [4,0], [5,0],
            ]
        , dtype=torch.float32); self.patch_all.append(patch_15)
        patch_16 = torch.tensor(
            [[5,0], [4,0], [3,0], [2,0], [1,0], [0,0],
            [5,1], [4,1], [3,1], [2,1], [1,1], [0,1],
            [5,2], [4,2], [3,2], [2,2], [1,2], [0,2],
            [5,3], [4,3], [3,3], [2,3], [1,3], [0,3],
            ]
        , dtype=torch.float32); self.patch_all.append(patch_16)
        # patch_16 = patch_15.clone(); self.patch_all.append(patch_16)
        patch_17 = patch_16.clone(); self.patch_all.append(patch_17)

if __name__=='__main__':
    csv_file = pd.read_csv(csv_file)
    AHTactilePlayer(csv_file, plot_Hz)