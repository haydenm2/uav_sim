import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from geometry.geometry import Rx_p, Ry_p, Rz_p

class UAV_simulator:
    def __init__(self, x0=np.array([[0], [0], [0]]), rpy=np.array([np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)]), follow=True):
        # UAV and target conditions
        self.x = x0     # UAV state
        self.rpy = rpy   # UAV attitude (roll, pitch, yaw)

        # General vectors and plotting variables
        self.line_scale = 5
        self.e1 = np.array([[1], [0], [0]])  # basis vector 1
        self.e2 = np.array([[0], [1], [0]])  # basis vector 2
        self.e3 = np.array([[0], [0], [1]])  # basis vector 3
        self.plt_follow_window = 20
        plt_range = 50
        self.follow = follow

        # ----------------------- Initialize 3D Simulation Plot -----------------------

        # Setup plotting base for 3D sim
        fig = plt.figure(1)
        self.ax = fig.add_subplot(111, projection='3d')    # plot axis handle
        if self.follow:
            self.ax.set_xlim(-self.plt_follow_window/2 + self.x[0], self.plt_follow_window/2 + self.x[0])  # plotting x limit
            self.ax.set_ylim(-self.plt_follow_window/2 + self.x[1], self.plt_follow_window/2 + self.x[1])  # plotting y limit
            self.ax.set_zlim(-self.plt_follow_window/2 + self.x[2], self.plt_follow_window/2 + self.x[2])  # plotting z limit
        else:
            self.ax.set_xlim(-plt_range, plt_range)      # plotting x limit
            self.ax.set_ylim(-plt_range, plt_range)      # plotting y limit
            self.ax.set_zlim(-plt_range, plt_range)      # plotting z limit
        self.ax.set_title('3D Sim')      # plotting title
        self.ax.set_xlabel('x')      # plotting x axis label
        self.ax.set_ylabel('y')      # plotting y axis label
        self.ax.set_zlabel('z')      # plotting z axis label
        self.ax.grid(True)      # plotting show grid lines

        # Define initial rotation matrices
        self.R_iv = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]])      # rotation matrix from inertial to vehicle frame
        self.R_vb = Rz_p(self.rpy[2]) @ Ry_p(self.rpy[1]) @ Rx_p(self.rpy[0])      # rotation matrix from vehicle to body frame
        self.R_ib = self.R_vb @ self.R_iv

        # Rotate animation lines and points back out to common inertial frame for plotting
        lx = self.R_iv.transpose() @ self.R_vb.transpose() @ np.hstack((np.zeros((3, 1)), self.e1)) * self.line_scale + self.x
        ly = self.R_iv.transpose() @ self.R_vb.transpose() @ np.hstack((np.zeros((3, 1)), self.e2)) * self.line_scale + self.x
        lz = self.R_iv.transpose() @ self.R_vb.transpose() @ np.hstack((np.zeros((3, 1)), self.e3)) * self.line_scale + self.x
        lx_i = np.hstack((np.zeros((3, 1)), self.e1)) * self.line_scale
        ly_i = np.hstack((np.zeros((3, 1)), self.e2)) * self.line_scale
        lz_i = np.hstack((np.zeros((3, 1)), self.e3)) * self.line_scale

        # Plotting handles
        self.plx = self.ax.plot3D(lx[0, :], lx[1, :], lx[2, :], 'r-')  # plot body x-axis
        self.ply = self.ax.plot3D(ly[0, :], ly[1, :], ly[2, :], 'g-')  # plot body y-axis
        self.plz = self.ax.plot3D(lz[0, :], lz[1, :], lz[2, :], 'b-')  # plot body z-axis
        self.plx_i = self.ax.plot3D(lx_i[0, :], lx_i[1, :], lx_i[2, :], 'm-')  # plot inertial x-axis
        self.ply_i = self.ax.plot3D(ly_i[0, :], ly_i[1, :], ly_i[2, :], 'y-')  # plot inertial y-axis
        self.plz_i = self.ax.plot3D(lz_i[0, :], lz_i[1, :], lz_i[2, :], 'c-')  # plot inertial z-axis

        self.UpdateSim()

    # update UAV position
    def UpdateX(self, x):
        self.x = x.reshape(-1, 1)
        self.UpdateSim()

    # update UAV orientation
    def UpdateRPY(self, rpy):
        self.rpy = rpy
        self.R_vb = Rz_p(self.rpy[2]) @ Ry_p(self.rpy[1]) @ Rx_p(self.rpy[0])
        self.R_ib = self.R_vb @ self.R_iv
        self.UpdateSim()

    # updates 3d simulation
    def UpdateSim(self):
        # Rotate animation lines and points back out to common inertial frame for plotting
        lx = np.hstack((self.x, self.R_iv.transpose() @ self.R_vb.transpose() @ np.hstack((np.zeros((3, 1)), self.e1)) * self.line_scale + self.x))
        ly = np.hstack((self.x, self.R_iv.transpose() @ self.R_vb.transpose() @ np.hstack((np.zeros((3, 1)), self.e2)) * self.line_scale + self.x))
        lz = np.hstack((self.x, self.R_iv.transpose() @ self.R_vb.transpose() @ np.hstack((np.zeros((3, 1)), self.e3)) * self.line_scale + self.x))

        # Plotting handles data updates
        self.plx[0].set_data_3d(lx[0, :], lx[1, :], lx[2, :])  # update plot body x-axis
        self.ply[0].set_data_3d(ly[0, :], ly[1, :], ly[2, :])  # update plot body y-axis
        self.plz[0].set_data_3d(lz[0, :], lz[1, :], lz[2, :])  # update plot body z-axis

        if self.follow:
            self.ax.set_xlim(-self.plt_follow_window / 2 + self.x[0], self.plt_follow_window / 2 + self.x[0])  # update plot x limit
            self.ax.set_ylim(-self.plt_follow_window / 2 + self.x[1], self.plt_follow_window / 2 + self.x[1])  # update plot y limit
            self.ax.set_zlim(-self.plt_follow_window / 2 + self.x[2], self.plt_follow_window / 2 + self.x[2])  # update plot z limit

        # self.ax.autoscale()
        plt.pause(0.01)
