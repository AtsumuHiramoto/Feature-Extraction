import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import axes3d, Axes3D
import math
import tkinter
import time
matplotlib.use('TkAgg')

def main():
    coordinate_file = "ALLEGRO HAND TAXEL POSITIONS.csv"
    posture = "up"
    # posture = "down"
    scm = SensorCoordinateManager(coordinate_file=coordinate_file, posture=posture)

    # home position
    joint_angle = torch.tensor([[0,0,0,0],
                   [0,0,0,0],
                   [0,0,0,0],
                   [0,0,0,0]])
    joint_angle = joint_angle * math.pi /180
    start_time = time.time()
    trans_joint, trans_tac = scm.transform(joint_angle=joint_angle, plot_image=False)
    print("process time: {}".format(time.time()-start_time))
    trans_joint, trans_tac = scm.transform(joint_angle=joint_angle, plot_image=True)

    # other position
    joint_angle = torch.tensor([[0,10,10,10],
                   [0,90,0,0],
                   [-90,10,20,30],
                   [-90,-90,0,-90]])
    joint_angle = joint_angle * math.pi /180
    start_time = time.time()
    trans_joint, trans_tac = scm.transform(joint_angle=joint_angle, plot_image=False)
    print("process time: {}".format(time.time()-start_time))
    trans_joint, trans_tac = scm.transform(joint_angle=joint_angle, plot_image=True)

class SensorCoordinateManager(object):
    def __init__(self, coordinate_file, posture="up"):
        self.posture = posture
        home_coordinates_df = pd.read_csv(coordinate_file)
        self.tac_coordinate = home_coordinates_df[["X","Y","Z"]].values # [368,3]
        self.tac_coordinate = torch.from_numpy(self.tac_coordinate)
        self.set_param()
        self.set_posture()
        # self.plot(self.joint_coordinate, self.tac_coordinate)
    def set_posture(self):
        if self.posture=="up":
            self.joint_coordinate[:,:,[0,2]] = -self.joint_coordinate[:,:,[0,2]]
            self.tac_coordinate[:,[0,2]] = -self.tac_coordinate[:,[0,2]]
            self.link_coordinate[:,:,[0,2]] = -self.link_coordinate[:,:,[0,2]]

    def transform(self, joint_angle, plot_image=False):
        tac_coordinate_trans = self.tac_coordinate.clone()
        joint_coordinate_trans = self.joint_coordinate.clone()
        joint_angle = joint_angle.clone()
        if self.posture=="up":
            calib_direction = 1.0
        if self.posture=="down":
            joint_angle[:3,1:] = -joint_angle[:3,1:]
            joint_angle[3,1] = -joint_angle[3,1]
            calib_direction = -1.0
        for joint_index, joint in enumerate(["Index", "Middle", "Little", "Thumb"]):
            if joint=="Thumb":
                # continue
                matrix_calib = self.make_matrix(axis="z", theta=calib_direction*self.calib_z_angle[2])
                matrix_base = self.make_matrix(axis="y", theta=joint_angle[joint_index][0], link_xyz=self.link_coordinate[joint_index][0])                
                matrix_1 = self.make_matrix(axis="x", theta=joint_angle[joint_index][1], link_xyz=self.link_coordinate[joint_index][1])
                matrix_2 = self.make_matrix(axis="y", theta=joint_angle[joint_index][2], link_xyz=self.link_coordinate[joint_index][2])
                matrix_3 = self.make_matrix(axis="y", theta=joint_angle[joint_index][3], link_xyz=self.link_coordinate[joint_index][3])
                
                matrix_j1 = torch.matmul(matrix_base, matrix_1)
                matrix_j1 = torch.matmul(matrix_j1, matrix_calib)
                matrix_j2 = torch.matmul(matrix_j1, matrix_2)
                matrix_j3 = torch.matmul(matrix_j2, matrix_3)
            else:
                matrix_base = self.make_matrix(axis="y", theta=joint_angle[joint_index][0], link_xyz=self.link_coordinate[joint_index][0])        
                matrix_1 = self.make_matrix(axis="x", theta=joint_angle[joint_index][1], link_xyz=self.link_coordinate[joint_index][1])
                matrix_2 = self.make_matrix(axis="x", theta=joint_angle[joint_index][2], link_xyz=self.link_coordinate[joint_index][2])
                matrix_3 = self.make_matrix(axis="x", theta=joint_angle[joint_index][3], link_xyz=self.link_coordinate[joint_index][3])

                if joint=="Middle":
                    matrix_j1 = torch.matmul(matrix_base, matrix_1)
                else:
                    if joint=="Index":
                        matrix_calib = self.make_matrix(axis="z", theta=calib_direction*self.calib_z_angle[0])
                    if joint=="Little":
                        matrix_calib = self.make_matrix(axis="z", theta=calib_direction*self.calib_z_angle[1])
                    matrix_j1 = torch.matmul(matrix_base, matrix_calib)
                    matrix_j1 = torch.matmul(matrix_j1, matrix_1)
                matrix_j2 = torch.matmul(matrix_j1, matrix_2)
                matrix_j3 = torch.matmul(matrix_j2, matrix_3)

            # update tactiles
            for patch_index, patch_range in enumerate(self.hand_sensor_layout[joint]):
                if joint=="Thumb":
                    for sensor_num in patch_range:
                        if patch_index==2:
                            tac_matrix_index = self.make_matrix(link_xyz=self.tac_coordinate[sensor_num] - self.joint_coordinate[joint_index][2])
                            tac_coordinate_trans[sensor_num] = torch.matmul(matrix_j2, tac_matrix_index)[:3,3]
                        elif patch_index==0 or patch_index==1: # fingertip
                            tac_matrix_index = self.make_matrix(link_xyz=self.tac_coordinate[sensor_num] - self.joint_coordinate[joint_index][3])
                            tac_coordinate_trans[sensor_num] = torch.matmul(matrix_j3, tac_matrix_index)[:3,3]
                else:
                    for sensor_num in patch_range:
                        if patch_index==2 or patch_index==3: # base
                            tac_matrix_index = self.make_matrix(link_xyz=self.tac_coordinate[sensor_num] - self.joint_coordinate[joint_index][1])
                            tac_coordinate_trans[sensor_num] = torch.matmul(matrix_j1, tac_matrix_index)[:3,3]
                        elif patch_index==1:
                            tac_matrix_index = self.make_matrix(link_xyz=self.tac_coordinate[sensor_num] - self.joint_coordinate[joint_index][2])
                            tac_coordinate_trans[sensor_num] = torch.matmul(matrix_j2, tac_matrix_index)[:3,3]
                        elif patch_index==0: # fingertip
                            tac_matrix_index = self.make_matrix(link_xyz=self.tac_coordinate[sensor_num] - self.joint_coordinate[joint_index][3])
                            tac_coordinate_trans[sensor_num] = torch.matmul(matrix_j3, tac_matrix_index)[:3,3]

            # update joints
            joint_coordinate_trans[joint_index][1] = matrix_j1[:3,3]
            joint_coordinate_trans[joint_index][2] = matrix_j2[:3,3]
            joint_coordinate_trans[joint_index][3] = matrix_j3[:3,3]

        if plot_image==True:
            self.plot(joint_coordinate_trans, tac_coordinate_trans)

        return joint_coordinate_trans, tac_coordinate_trans
            
    def make_matrix(self, axis=None, theta=None, link_xyz=torch.tensor([0,0,0])):
        link_x, link_y, link_z = link_xyz.tolist()
        if axis=="x":
            matrix = torch.tensor([
                [1, 0, 0, link_x],
                [0, math.cos(theta), -math.sin(theta), link_y],
                [0, math.sin(theta), math.cos(theta), link_z],
                [0, 0, 0, 1]
            ])
        elif axis=="y":
            matrix = torch.tensor([
                [math.cos(theta), 0, math.sin(theta), link_x],
                [0, 1, 0, link_y],
                [-math.sin(theta), 0, math.cos(theta), link_z],
                [0, 0, 0, 1]
            ])
        elif axis=="z":
            matrix = torch.tensor([
                [math.cos(theta), -math.sin(theta), 0, link_x],
                [math.sin(theta), math.cos(theta), 0, link_y],
                [0, 0, 1, link_z],
                [0, 0, 0, 1]
            ])
        else:
            matrix = torch.tensor([
                [1, 0, 0, link_x],
                [0, 1, 0, link_y],
                [0, 0, 1, link_z],
                [0, 0, 0, 1]
            ])
        return matrix
    def plot(self, joint_coordinates, tac_coordinates):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        if self.posture=="up":
            plt.xlim(-100, 250)
            plt.ylim(-50, 300)
            ax.set_zlim(-100, 150)
        elif self.posture=="down":
            plt.xlim(-250, 100)
            plt.ylim(-50, 300)
            ax.set_zlim(-200, 50)
        joint_coordinates = joint_coordinates.view(-1, 3)
        ax.scatter(joint_coordinates[:,0],
                   joint_coordinates[:,1],
                   joint_coordinates[:,2],
                   color="b")
        # ax.scatter(joint_coordinates[:4,0],
        #            joint_coordinates[:4,1],
        #            joint_coordinates[:4,2],
        #            color="r")
        ax.scatter(tac_coordinates[:,0],
                   tac_coordinates[:,1],
                   tac_coordinates[:,2],
                   color="g")

        ax.set_xlabel('$X$', fontsize=20, rotation=150)
        ax.set_ylabel('$Y$', fontsize=20, rotation=150)
        ax.set_zlabel('$Z$', fontsize=20, rotation=150)
        plt.show()

    def set_param(self):
        joint_c = torch.tensor([
            [[-43.773, 94.151, 13.300], [-45.446, 113.277, 13.300], [-50.153, 167.072, 13.300], [-53.848, 209.311, 13.300]],
            [[0, 96.4, 13.300], [0, 115.6, 13.300], [0, 169.6, 13.300], [0, 212.0, 13.300]],
            [[44.104, 97.936, 13.300], [45.446, 113.277, 13.300], [50.153, 167.072, 13.300], [53.848, 209.311, 13.300]],
            [[-19.331, 48.609, 31.500], [-60.79, 17.877, 26.500], [-76.131, 16.535, 26.500], [-127.336, 12.336, 26.500]]])
        self.calib_z_angle = torch.tensor([math.atan2(joint_c[0][3][0] - joint_c[0][0][0], joint_c[0][3][1] - joint_c[0][0][1]),
                                        #    math.atan2(joint_c[1][3][0] - joint_c[1][0][0], joint_c[1][3][1] - joint_c[1][0][1]),
                                           math.atan2(joint_c[2][3][0] - joint_c[2][0][0], joint_c[2][3][1] - joint_c[2][0][1]),
                                           math.atan2(joint_c[0][3][0] - joint_c[0][1][0], joint_c[0][3][1] - joint_c[0][1][1])])
        self.joint_coordinate = joint_c
        # initial pseudo link
        self.make_link(joint_c)

        self.hand_sensor_layout = {
            'Index' : [[i for i in range(30)], [i for i in range(30, 46)], [i for i in range(46, 62)], [i for i in range(62, 78)]], # from fingertip to base
            'Middle' : [[i for i in range(78, 108)], [i for i in range(108, 124)], [i for i in range(124, 140)], [i for i in range(140, 156)]],
            'Little' : [[i for i in range(156, 186)], [i for i in range(186, 202)], [i for i in range(202, 218)], [i for i in range(218, 234)]],
            'Thumb' : [[i for i in range(234, 264)], [i for i in range(264, 280)], [i for i in range(280, 296)]]
                        }
        self.calibrate_home_coordinates()
        # update link
        self.make_link(self.joint_coordinate)

    def make_link(self, joint_c):
        self.link_coordinate = torch.stack([
            joint_c[0][0], joint_c[0][1] - joint_c[0][0], joint_c[0][2] - joint_c[0][1], joint_c[0][3] - joint_c[0][2],
            joint_c[1][0], joint_c[1][1] - joint_c[1][0], joint_c[1][2] - joint_c[1][1], joint_c[1][3] - joint_c[1][2],
            joint_c[2][0], joint_c[2][1] - joint_c[2][0], joint_c[2][2] - joint_c[2][1], joint_c[2][3] - joint_c[2][2],
            joint_c[3][0], joint_c[3][1] - joint_c[3][0], joint_c[3][2] - joint_c[3][1], joint_c[3][3] - joint_c[3][2]
        ]).view(4,4,3)

    def calibrate_home_coordinates(self):
        #index
        matrix_calib_index = self.make_matrix(axis="z", theta=self.calib_z_angle[0], link_xyz=self.link_coordinate[0][0])
        for i in range(4):
            joint_matrix_index = self.make_matrix(link_xyz=self.joint_coordinate[0][i] - self.link_coordinate[0][0])
            self.joint_coordinate[0][i] = torch.matmul(matrix_calib_index, joint_matrix_index)[:3,3]
        for i in range(0, 78):
            tac_matrix_index = self.make_matrix(link_xyz=self.tac_coordinate[i] - self.link_coordinate[0][0])
            self.tac_coordinate[i] = torch.matmul(matrix_calib_index, tac_matrix_index)[:3,3]
        # little
        matrix_calib_little = self.make_matrix(axis="z", theta=self.calib_z_angle[1], link_xyz=self.link_coordinate[2][0])
        for i in range(4):
            joint_matrix_little = self.make_matrix(link_xyz=self.joint_coordinate[2][i] - self.link_coordinate[2][0])
            self.joint_coordinate[2][i] = torch.matmul(matrix_calib_little, joint_matrix_little)[:3,3]
        for i in range(156, 234):
            tac_matrix_little = self.make_matrix(link_xyz=self.tac_coordinate[i] - self.link_coordinate[2][0])
            self.tac_coordinate[i] = torch.matmul(matrix_calib_little, tac_matrix_little)[:3,3]
        # thumb
        matrix_calib_thumb = self.make_matrix(axis="z", theta=self.calib_z_angle[2], link_xyz=self.link_coordinate[3][1])
        for i in range(1,4):
            joint_matrix_thumb = self.make_matrix(link_xyz=self.joint_coordinate[3][i] - self.link_coordinate[3][1])
            self.joint_coordinate[3][i] = torch.matmul(matrix_calib_thumb, joint_matrix_thumb)[:3,3]
        for i in range(234, 296):
            tac_matrix_thumb = self.make_matrix(link_xyz=self.tac_coordinate[i] - self.link_coordinate[3][1])
            self.tac_coordinate[i] = torch.matmul(matrix_calib_thumb, tac_matrix_thumb)[:3,3]

if __name__=="__main__":
    main()
