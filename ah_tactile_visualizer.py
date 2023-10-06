import rospy
from sensor_msgs.msg import JointState
from datetime import datetime
# from allegro_hand_description.msg import ah_msg_cli, ah_msg_key
from std_srvs.srv import SetBool
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Float32MultiArray
from geometry_msgs.msg import Pose
#  from allegro_hand_description.msg import ah_node_state
import numpy as np
import os
import pandas as pd
import copy
from abc import ABCMeta, abstractmethod
import roslib
from multiprocessing.connection import Client
import threading
from time import sleep
import time
import csv
import psutil
import cv2
import pdb
import matplotlib.pyplot as plt
import torch

script_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))
package_dir = roslib.packages.get_pkg_dir("allegro_hand_client")
record_data_dir = os.path.join(package_dir, "record_data")

class AHTactileVisualizer(object):
    def __init__(self, record_rate=100, tactile_ip='127.0.0.1', tactile_ports=[5007, 5008]):
        rospy.init_node('tactile_visualizer')
        self.__robot_state = {
            'client_time': rospy.Time.now(),
            'time_stamp': rospy.Time.now(),
            'arm_time_stamp': rospy.Time.now(),
            'joint': JointState().position,
            'desired_joint': JointState().position,
            'torque': JointState().effort,
            'desired_torque': JointState().effort,
            'arm_joint': JointState().position,
            'arm_torque': JointState().effort,
            'tactile1': Float32MultiArray().data,
            'tactile2': Float32MultiArray().data,
            'tactile3': Float32MultiArray().data,
            'tactile4': Float32MultiArray().data,
            'tactile5': Float32MultiArray().data,
            'SixAxis_0_force': Pose().position,
            'SixAxis_1_force': Pose().position,
            'SixAxis_2_force': Pose().position,
            'SixAxis_3_force': Pose().position,
            'SixAxis_0_torque': Pose().orientation,
            'SixAxis_1_torque': Pose().orientation,
            'SixAxis_2_torque': Pose().orientation,
            'SixAxis_3_torque': Pose().orientation,
            'image': Int32MultiArray().data
        }
        rospy.Subscriber('/allegroHand_0/joint_states',
                         JointState, callback=self.__robot_states_handler,
                         queue_size=1, tcp_nodelay=True)
        
        # rospy.Subscriber('/allegroHand/desired_joint_states',
        #                  JointState, callback=self.__desired_robot_states_handler,
        #                  queue_size=1, tcp_nodelay=True)

        rospy.Subscriber('/allegroHand/tactile_states_index',
                         Float32MultiArray, callback=self.__tactile_states_handler_1,
                         queue_size=1, tcp_nodelay=True)

        rospy.Subscriber('/allegroHand/tactile_states_middle',
                         Float32MultiArray, callback=self.__tactile_states_handler_2,
                         queue_size=1, tcp_nodelay=True)

        rospy.Subscriber('/allegroHand/tactile_states_little',
                         Float32MultiArray, callback=self.__tactile_states_handler_3,
                         queue_size=1, tcp_nodelay=True)

        rospy.Subscriber('/allegroHand/tactile_states_thumb',
                         Float32MultiArray, callback=self.__tactile_states_handler_4,
                         queue_size=1, tcp_nodelay=True)

        rospy.Subscriber('/allegroHand/tactile_states_palm',
                         Float32MultiArray, callback=self.__tactile_states_handler_5,
                         queue_size=1, tcp_nodelay=True)

        self.fig = plt.figure(figsize=(10,10))
        # self.fig = plt.figure(figsize=(20,20))
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")
        # plt.xlim(0,500)
        # plt.ylim(0,500)
        # plt.xlim(-500,1000)
        # plt.ylim(-500,1000)
        self.set_patch_arrangement()

        rospy.sleep(1)
        self.run_loop()

    def __robot_states_handler(self, msg):
        self.__robot_state["torque"] = msg.effort
        self.__robot_state["joint"] = msg.position
        self.__robot_state["time_stamp"] = msg.header.stamp
    def __tactile_states_handler_1(self, msg):
        # rospy.logwarn("tactile1" )
        self.__robot_state["tactile1"] = msg.data

    def __tactile_states_handler_2(self, msg):
        # rospy.logwarn("tactile2" )
        self.__robot_state["tactile2"] = msg.data

    def __tactile_states_handler_3(self, msg):
        # rospy.logwarn("tactile3" )
        self.__robot_state["tactile3"] = msg.data

    def __tactile_states_handler_4(self, msg):
        self.__robot_state["tactile4"] = msg.data

    def __tactile_states_handler_5(self, msg):
        self.__robot_state["tactile5"] = msg.data
    def run_loop(self,):
        while not rospy.is_shutdown():
            self.__robot_state['client_time'] = \
                rospy.Time.now().to_sec()
            self.__robot_state['time_stamp'] = \
                rospy.Time.now().to_sec()
            self.__robot_state_log = [copy.copy(self.__robot_state)]
            # print(self.__robot_state)
            tmp = np.zeros(len(self.__robot_state_log))
            for i, item in enumerate(self.__robot_state_log):
                tmp[i] = item['client_time']
            client_time_df = pd.DataFrame(tmp,
                                        columns=['ClientTimeStamp'])

            # Format time stamp log
            tmp = np.zeros(len(self.__robot_state_log))
            for i, item in enumerate(self.__robot_state_log):
                # tmp[i] = item["time_stamp"].to_sec()
                tmp[i] = item["time_stamp"]
            time_stamp_df = pd.DataFrame(tmp,
                                        columns=["ServerTimeStamp"])

            # Format joint angle log
            header_str = []
            for j in range(4):  # num of finger
                for k in range(4):  # num of finger joint
                    header_str.append('JointF'+str(j)+'J'+str(k))
            tmp = np.zeros((len(self.__robot_state_log),
                            len(header_str)))
            for i, item in enumerate(self.__robot_state_log):
                if len(item["joint"]) ==0:
                    break
                tmp[i] = item["joint"]
            joint_df = pd.DataFrame(tmp,
                                    columns=header_str)

            # Format desired joint angle log
            header_str = []
            for j in range(4):  # num of finger
                for k in range(4):  # num of finger joint
                    header_str.append('DesJointF'+str(j)+'J'+str(k))
            tmp = np.zeros((len(self.__robot_state_log),
                            len(header_str)))
            for i, item in enumerate(self.__robot_state_log):
                if len(item["desired_joint"]) ==0:
                    break
                tmp[i] = item["desired_joint"]
            des_joint_df = pd.DataFrame(tmp,
                                        columns=header_str)

            # Format joint torque log
            header_str = []
            for j in range(4):  # num of finger
                for k in range(4):  # num of finger joint
                    header_str.append('TorqueF'+str(j)+'J'+str(k))
            tmp = np.zeros((len(self.__robot_state_log),
                            len(header_str)))
            for i, item in enumerate(self.__robot_state_log):
                if len(item["torque"]) ==0:
                    break
                tmp[i] = item["torque"]
            torque_df = pd.DataFrame(tmp,
                                    columns=header_str)

            # Format desired joint torque log
            header_str = []
            for j in range(4):  # num of finger
                for k in range(4):  # num of finger joint
                    header_str.append('DesTorqueF'+str(j)+'J'+str(k))
            tmp = np.zeros((len(self.__robot_state_log),
                            len(header_str)))
            for i, item in enumerate(self.__robot_state_log):
                if len(item["desired_torque"]) ==0:
                    break
                tmp[i] = item["desired_torque"]
            des_torque_df = pd.DataFrame(tmp,
                                        columns=header_str)
            # Format arm joint angle log
            header_str = []
            for j in range(7):  # num of arm joint
                header_str.append('ArmJoint'+str(j))
            tmp = np.zeros((len(self.__robot_state_log),
                            len(header_str)))
            for i, item in enumerate(self.__robot_state_log):
                if len(item["arm_joint"]) ==0:
                    break
                tmp[i] = item["arm_joint"]
            arm_joint_df = pd.DataFrame(tmp,
                                    columns=header_str)

            # Format joint torque log
            header_str = []
            for j in range(7):  # num of arm joint
                header_str.append('ArmTorque'+str(j))
            tmp = np.zeros((len(self.__robot_state_log),
                            len(header_str)))
            for i, item in enumerate(self.__robot_state_log):
                if len(item["arm_torque"]) ==0:
                    break
                tmp[i] = item["arm_torque"]
            arm_torque_df = pd.DataFrame(tmp,
                                    columns=header_str)
            
            # Format image log
            # tmp = np.zeros((len(self.__robot_state_log),
            #                     len(self.__robot_state["image"])))
            # tmp = []
            # for i, item in enumerate(self.__robot_state_log):
            #     arr = np.asarray(item["image"], dtype=np.uint8)
            #     image_encode = cv2.imdecode(arr, -1)
            #     tmp.append(image_encode)
            # image_df = tmp
            
            
            ####### tactile ########
            # Format tactile1 log
            header_str = []
            # index_tip
            for j in range(1):  # board
                for k in range(30):  # taxel
                    header_str.append('IndexTip_Tactile'+'B'+str(j).zfill(2)+'C'+str(k).zfill(2)+'X')
                    header_str.append('IndexTip_Tactile'+'B'+str(j).zfill(2)+'C'+str(k).zfill(2)+'Y')
                    header_str.append('IndexTip_Tactile'+'B'+str(j).zfill(2)+'C'+str(k).zfill(2)+'Z')
            # index_phalange
            for j in range(3):  # board
                for k in range(16):  # taxel
                    header_str.append('IndexPhalange_Tactile'+'B'+str(j).zfill(2)+'C'+str(k).zfill(2)+'X')
                    header_str.append('IndexPhalange_Tactile'+'B'+str(j).zfill(2)+'C'+str(k).zfill(2)+'Y')
                    header_str.append('IndexPhalange_Tactile'+'B'+str(j).zfill(2)+'C'+str(k).zfill(2)+'Z')
            tmp = np.zeros((len(self.__robot_state_log,),
                            len(header_str)))
            for i, item in enumerate(self.__robot_state_log):
                if len(item["tactile1"]) ==0:
                    break
                tmp[i] = item["tactile1"]
            tactile1_df = pd.DataFrame(tmp,
                                    columns=header_str)

            # Format tactile2 log
            header_str = []
            # middle_tip
            for j in range(1):  # board
                for k in range(30):  # taxel
                    header_str.append('MiddleTip_Tactile'+'B'+str(j).zfill(2)+'C'+str(k).zfill(2)+'X')
                    header_str.append('MiddleTip_Tactile'+'B'+str(j).zfill(2)+'C'+str(k).zfill(2)+'Y')
                    header_str.append('MiddleTip_Tactile'+'B'+str(j).zfill(2)+'C'+str(k).zfill(2)+'Z')
            # middle_phalange
            for j in range(3):  # board
                for k in range(16):  # taxel
                    header_str.append('MiddlePhalange_Tactile'+'B'+str(j).zfill(2)+'C'+str(k).zfill(2)+'X')
                    header_str.append('MiddlePhalange_Tactile'+'B'+str(j).zfill(2)+'C'+str(k).zfill(2)+'Y')
                    header_str.append('MiddlePhalange_Tactile'+'B'+str(j).zfill(2)+'C'+str(k).zfill(2)+'Z')
            tmp = np.zeros((len(self.__robot_state_log,),
                            len(header_str)))
            for i, item in enumerate(self.__robot_state_log):
                if len(item["tactile2"]) ==0:
                    break
                tmp[i] = item["tactile2"]
            tactile2_df = pd.DataFrame(tmp,
                                    columns=header_str)

            # Format tactile3 log
            header_str = []
            # Little_tip
            for j in range(1):  # board
                for k in range(30):  # taxel
                    header_str.append('LittleTip_Tactile'+'B'+str(j).zfill(2)+'C'+str(k).zfill(2)+'X')
                    header_str.append('LittleTip_Tactile'+'B'+str(j).zfill(2)+'C'+str(k).zfill(2)+'Y')
                    header_str.append('LittleTip_Tactile'+'B'+str(j).zfill(2)+'C'+str(k).zfill(2)+'Z')
            # Little_phalange
            for j in range(3):  # board
                for k in range(16):  # taxel
                    header_str.append('LittlePhalange_Tactile'+'B'+str(j).zfill(2)+'C'+str(k).zfill(2)+'X')
                    header_str.append('LittlePhalange_Tactile'+'B'+str(j).zfill(2)+'C'+str(k).zfill(2)+'Y')
                    header_str.append('LittlePhalange_Tactile'+'B'+str(j).zfill(2)+'C'+str(k).zfill(2)+'Z')
            tmp = np.zeros((len(self.__robot_state_log,),
                            len(header_str)))
            for i, item in enumerate(self.__robot_state_log):
                if len(item["tactile3"]) ==0:
                    break
                tmp[i] = item["tactile3"]
            tactile3_df = pd.DataFrame(tmp,
                                    columns=header_str)

            # Format tactile4 log
            header_str = []
            # thumb_tip
            for j in range(1):  # board
                for k in range(30):  # taxel
                    header_str.append('ThumbTip_Tactile'+'B'+str(j).zfill(2)+'C'+str(k).zfill(2)+'X')
                    header_str.append('ThumbTip_Tactile'+'B'+str(j).zfill(2)+'C'+str(k).zfill(2)+'Y')
                    header_str.append('ThumbTip_Tactile'+'B'+str(j).zfill(2)+'C'+str(k).zfill(2)+'Z')
            # thumb_phlange
            for j in range(2):  # board
                for k in range(16):  # taxel
                    header_str.append('ThumbPhalange_Tactile'+'B'+str(j).zfill(2)+'C'+str(k).zfill(2)+'X')
                    header_str.append('ThumbPhalange_Tactile'+'B'+str(j).zfill(2)+'C'+str(k).zfill(2)+'Y')
                    header_str.append('ThumbPhalange_Tactile'+'B'+str(j).zfill(2)+'C'+str(k).zfill(2)+'Z')
            tmp = np.zeros((len(self.__robot_state_log,),
                            len(header_str)))
            for i, item in enumerate(self.__robot_state_log):
                if len(item["tactile4"]) ==0:
                    break
                tmp[i] = item["tactile4"]
            tactile4_df = pd.DataFrame(tmp,
                                    columns=header_str)

            # Format tactile5 log
            header_str = []
            # palm
            for j in range(3):  # board
                for k in range(24):  # taxel
                    header_str.append('Palm_Tactile'+'B'+str(j).zfill(2)+'C'+str(k).zfill(2)+'X')
                    header_str.append('Palm_Tactile'+'B'+str(j).zfill(2)+'C'+str(k).zfill(2)+'Y')
                    header_str.append('Palm_Tactile'+'B'+str(j).zfill(2)+'C'+str(k).zfill(2)+'Z')
            tmp = np.zeros((len(self.__robot_state_log,),
                            len(header_str)))
            for i, item in enumerate(self.__robot_state_log):
                if len(item["tactile5"]) ==0:
                    break
                tmp[i] = item["tactile5"]
            tactile5_df = pd.DataFrame(tmp,
                                    columns=header_str)

            # merge_df = pd.concat(
            #     [client_time_df, time_stamp_df, joint_df, des_joint_df, torque_df, arm_joint_df, arm_torque_df], axis=1)
            # merge_df = pd.concat(
            #     [client_time_df, time_stamp_df, joint_df, des_joint_df, torque_df, arm_joint_df, arm_torque_df, tactile_test_df], axis=1)
            # merge_df = pd.concat(
            #     [client_time_df, time_stamp_df, joint_df, 
            #      des_joint_df, torque_df, 
            #      tactile1_df, tactile2_df, tactile3_df,
            #      tactile4_df, tactile5_df], axis=1)
            # print(tactile2_df)#ueno debug
            merge_df = pd.concat(
                [client_time_df, time_stamp_df, joint_df, des_joint_df, 
                torque_df, des_torque_df, arm_joint_df, arm_torque_df, 
                tactile1_df, tactile2_df, tactile3_df,
                tactile4_df, tactile5_df], axis=1)
            # merge_df = pd.concat(
            #     [client_time_df, time_stamp_df, joint_df, 
            #      des_joint_df, torque_df, arm_joint_df, arm_torque_df, ], axis=1)
            # end_msg = ah_msg_key()
            # end_msg.data = "end_rec"
            # self.__lib_cmd_key_publisher.publish(end_msg)

            # self.__wait_for_filename_postfix()

            # self.__write_csv(merge_df)
            
            # self.__write_image(image_df)
            self.simple_tactile_viz(merge_df)
            
            # import ipdb; ipdb.set_trace()
    def simple_tactile_viz(self, merge_df):
        # import ipdb; ipdb.set_trace()
        tac_df = merge_df.filter(like='Tactile',axis=1)
        self.finger_name = ['Index', 'Middle', 'Little', 'Palm']
        self.patch_name = ['IndexTip_TactileB00', 'IndexPhalange_TactileB00', 'IndexPhalange_TactileB01', 'IndexPhalange_TactileB02',
                           'MiddleTip_TactileB00', 'MiddlePhalange_TactileB00', 'MiddlePhalange_TactileB01', 'MiddlePhalange_TactileB02',
                           'LittleTip_TactileB00', 'LittlePhalange_TactileB00', 'LittlePhalange_TactileB01', 'LittlePhalange_TactileB02',
                           'ThumbTip_TactileB00', 'ThumbPhalange_TactileB00', 'ThumbPhalange_TactileB01',
                           'Palm_TactileB00', 'Palm_TactileB01', 'Palm_TactileB02']

        # patch_one = True
        # if patch_one==True:
        #     patch_index = 2
        #     patch_df = (merge_df.filter(like=self.patch_name[patch_index],axis=1))
        #     self.plot(patch_df, patch_index)
        #     # self.plot_finger_phalange(patch_df, self.patch_1, self.patch_all[patch_index])

        patch_all = True
        if patch_all == True:
            # for patch_index in range(0,2):
                # patch_df = (merge_df.filter(like=self.patch_name[patch_index],axis=1))
            self.plot_all(merge_df)
            # plt.pause(0.001)
            # self.ax.remove()

    def plot(self, patch_df, patch_index):
        patch_arrangement = self.patch_all[patch_index]
        sensor_num = patch_arrangement.shape[0]
        scale = 200
        margin = 1000
        sensor_rate_x = 0.2
        sensor_rate_y = 0.2
        sensor_rate_z = 0.1
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")
        plt.xlim(scale - margin, scale*4 + margin)
        plt.ylim(scale - margin, scale*4 + margin)
        # import ipdb; ipdb.set_trace()
        # tactile_data = np.array(patch_df).reshape([16,3])
        tactile_data = np.array(patch_df).reshape([sensor_num,3])
        for s in range(sensor_num):
            marker_size = 20 + abs(int(tactile_data[s,2])*sensor_rate_z) # z axis value
            marker_color = 'g' if tactile_data[s,2] > 0  else 'b'
            ts_x = int(-tactile_data[s,1]*sensor_rate_x) # x axis translation
            ts_y = int(tactile_data[s,0]*sensor_rate_y) # y axis translation
            self.ax.plot([patch_arrangement[s][0]*scale + ts_x], [patch_arrangement[s][1]*scale + ts_y], '.', markersize=marker_size, color=marker_color)
            # print([(i+1)*scale], [(j+1)*scale])
            self.ax.plot([patch_arrangement[s][0]*scale], [patch_arrangement[s][1]*scale], '.', markersize=int(scale/50), color='r')            
        plt.pause(0.001)
        # self.ax.remove()
        self.ax.clear()

    def plot_all(self, merge_df):
        initial_position = [
            [14,25], [15,20], [15,15], [15,10], #index
            [7,25], [8,20], [8,15], [8,10], #middle
            [0,25], [1,20], [1,15], [1,10], #little
            [10,-12], [11,-5], [11,0], #thumb
            # [3.5,0], [3.5,5], [10.5,5] #palm
            [10.5,5], [3.5,5], [3.5,0] #palm
        ]
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
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")
        plt.xlim(-scale - 11.5*scale, scale*22 + 11.5*scale)
        plt.ylim(scale*(-13), scale*33)
        for patch_index in range(0,18):
            patch_df = (merge_df.filter(like=self.patch_name[patch_index],axis=1))
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
                # self.ax.plot([(patch_arrangement[s][0]+initial_position[patch_index][0])*scale + ts_x], 
                #             [(patch_arrangement[s][1]+initial_position[patch_index][1])*scale + ts_y], '.', markersize=marker_size, color=marker_color)
                # print([(i+1)*scale], [(j+1)*scale])
                # self.ax.plot([(patch_arrangement[s][0]+initial_position[patch_index][0])*scale],
                #             [(patch_arrangement[s][1]+initial_position[patch_index][1])*scale], '.', markersize=int(scale/50), color='r')
        # self.ax.set_prop_cycle(color=color_list)
        print(time.time()-start)
        self.ax.scatter(x_list,y_list, s=marker_size_list, marker='o', color=color_list)
        plt.pause(0.001)
        self.ax.remove()
        # plt.pause(0.001)
        # self.ax.remove()
    # def plot_all(self, patch_df, patch_index):
    #     initial_position = [
    #         [14,25], [15,20], [15,15], [15,10], #index
    #         [7,25], [8,20], [8,15], [8,10], #middle
    #         [0,25], [1,20], [1,15], [1,10], #little
    #         [11,0], [11,-5], [10,-12], #thumb
    #         [3.5,0], [3.5,5], [10.5,5] #palm
    #     ]
    #     patch_arrangement = self.patch_all[patch_index]
    #     if patch_index==12:
    #         patch_arrangement = patch_arrangement*(-1) + 5
    #     if patch_index==13 or patch_index==14:
    #         patch_arrangement = patch_arrangement*(-1) + 3
            
    #     # import ipdb; ipdb.set_trace()
    #     sensor_num = patch_arrangement.shape[0]
    #     scale = 200
    #     marker_size_lim = 10
    #     sensor_rate_x = 0.1
    #     sensor_rate_y = 0.1
    #     sensor_rate_z = 0.05
    #     self.ax = self.fig.add_subplot(111)
    #     self.ax.axis("off")
    #     plt.xlim(-scale - 11.5*scale, scale*22 + 11.5*scale)
    #     plt.ylim(scale*(-13), scale*33)

    #     # import ipdb; ipdb.set_trace()
    #     # tactile_data = np.array(patch_df).reshape([16,3])
    #     print(sensor_num)
    #     tactile_data = np.array(patch_df).reshape([sensor_num,3])
    #     for s in range(sensor_num):
    #         marker_size = marker_size_lim + abs(int(tactile_data[s,2])*sensor_rate_z) # z axis value
    #         marker_color = 'g' if tactile_data[s,2] > 0  else 'b'
    #         ts_x = int(-tactile_data[s,1]*sensor_rate_x) # x axis translation
    #         ts_y = int(tactile_data[s,0]*sensor_rate_y) # y axis translation
    #         self.ax.plot([(patch_arrangement[s][0]+initial_position[patch_index][0])*scale + ts_x], 
    #                      [(patch_arrangement[s][1]+initial_position[patch_index][1])*scale + ts_y], '.', markersize=marker_size, color=marker_color)
    #         # print([(i+1)*scale], [(j+1)*scale])
    #         self.ax.plot([(patch_arrangement[s][0]+initial_position[patch_index][0])*scale],
    #                      [(patch_arrangement[s][1]+initial_position[patch_index][1])*scale], '.', markersize=int(scale/50), color='r')

    #     # plt.pause(0.001)
    #     # self.ax.remove()

    def plot_finger_phalange(self, patch_df, patch_arrangement):
        scale = 200
        margin = 1000
        sensor_rate_x = 0.2
        sensor_rate_y = 0.2
        sensor_rate_z = 0.1
        self.ax = self.fig.add_subplot(111)
        self.ax.axis("off")
        plt.xlim(scale - margin, scale*4 + margin)
        plt.ylim(scale - margin, scale*4 + margin)
        import ipdb; ipdb.set_trace()
        tactile_data = np.array(patch_df).reshape([16,3])

        for i in range(4):
            for j in range(4):
                marker_size = 20 + abs(int(tactile_data[i*4+j,2])*sensor_rate_z) # z axis value
                marker_color = 'g' if tactile_data[i*4+j,2] > 0  else 'b'
                ts_x = int(tactile_data[i*4+j,0]*sensor_rate_x) # x axis translation
                ts_y = int(tactile_data[i*4+j,1]*sensor_rate_y) # y axis translation
                self.ax.plot([(i+1)*scale + ts_x], [(j+1)*scale + ts_y], '.', markersize=marker_size, color=marker_color)
                # print([(i+1)*scale], [(j+1)*scale])
                self.ax.plot([(i+1)*scale], [(j+1)*scale], '.', markersize=int(scale/50), color='r')
        plt.pause(0.001)
        self.ax.remove()
    
    def set_patch_arrangement(self):
        # self.patch_1 = np.array(
        #     [[3,0], [3,1], [3,2], [3,3],
        #     [2,0], [2,1], [2,2], [2,3],
        #     [1,0], [1,1], [1,2], [1,3],
        #     [0,0], [0,1], [0,2], [0,3],
        #     ])
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
    AHTactileVisualizer()
