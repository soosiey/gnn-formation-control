# -*- coding: utf-8 -*-

try:
    import cv2
    USE_CV2 = True
except ImportError:
    USE_CV2 = False

import operator
import math
from state import State
import numpy as np
import sim as vrep
from data import Data
from pointcloud import PointCloud
#import time
#import random


def saturate(dxp, dyp, dxypMax):
    dxyp = (dxp**2 + dyp**2)**0.5
    if dxyp > dxypMax:
        dxp = dxp / dxyp * dxypMax
        dyp = dyp / dxyp * dxypMax
    return dxp, dyp

class Robot():
    def __init__(self, scene,robot_num,if_train,expert_only,use_dagger,desired_distance,expert_velocity_adjust):

        ##### useful artribute

        self.index=None
        self.scene = scene
        self.l = 0.331
        self.nr = robot_num
        # state
        self.xi = State(0, 0, 0, self)
        self.xid = State(3, 0, 0, self)
        self.wheel_velocity_1 = 0
        self.wheel_velocity_2 = 0
        self.reachedGoal = False
        self.neighbors = []
        # Control parameters
        self.if_train=if_train
        self.expert_only=expert_only
        self.use_dagger=use_dagger
        self.expert_velocity_adjust=expert_velocity_adjust
        self.desired_distance=desired_distance
        self.p = 0.9
        self.control_vmax = 1.2
        self.control_vmin = 0.01
        self.accMax = 0.5
        ##### for future need
        self.pointCloud = PointCloud(self)
        # control parameters

        # Data to be recorded
        self.pose_list=[]
        self.velocity_list=[]


        self.recordData = False
        self.data = Data(self)
        self.v1Desired = 0
        self.v2Desired = 0
        self.v1Desirednn = 0
        self.v2Desirednn = 0
        # self.position_hist = []
        #### Robot's neighbor

        # self.leader = None # Only for data recording purposes

        self.ctrl1_sm = []
        self.ctrl2_sm = []
    def checkMove(self,hist,num = 1,thresh = .01):
        moving = False
        for i in range(1,len(hist)):
            if(abs(hist[i] - hist[i - 1]) > thresh):
                moving = True
        return moving
    #### Expert Controller
    def expert_control(self,omlist,index):
        ############## MODEL-BASED CONTROLLER (Most frequently used dynamics model) ##########
        ######################################################################################
        # For e-puk dynamics
        # Feedback linearization
        # v1: left wheel speed
        # v2: right wheel speed

        K3 = 1  # interaction between i and j

        # sort neighbors by distance

        # need all neighbors, but only diagonal of adjmatrix is 0 so it is okay
        if True:  # not hasattr(self, 'dictDistance'):
            self.dictDistance = dict()
            for j in range(len(self.scene.robots)):
                # if self.scene.adjMatrix[self.index, j] == 0:
                if self.index == j:
                    continue
                robot = self.scene.robots[j]  # neighbor
                self.dictDistance[j] = self.xi.distancepTo(robot.xi)
            self.listSortedDistance = sorted(self.dictDistance.items(),
                                             key=operator.itemgetter(1))
        # velocity in transformed space
        vxp = 0
        vyp = 0

        tauix = 0
        tauiy = 0
        # neighbors sorted by distances in descending order

        #### Use angle to get gabriel graph connections
        lsd = self.listSortedDistance
        jList = []
        for i in range(len(lsd)):
            connected = True
            for k in range(len(lsd)):
                if i == k:
                    continue
                ri = lsd[i][0]
                rk = lsd[k][0]
                di = np.array([self.xi.xp - self.scene.robots[rk].xi.xp, self.xi.yp - self.scene.robots[rk].xi.yp])
                dj = np.array([self.scene.robots[ri].xi.xp - self.scene.robots[rk].xi.xp,
                               self.scene.robots[ri].xi.yp - self.scene.robots[rk].xi.yp])
                c = np.dot(di, dj) / (np.linalg.norm(di) * np.linalg.norm(dj))
                angle = np.degrees(np.arccos(c))
                if (angle >= 90 and i != k):
                    connected = False
            if (connected):
                jList.append(lsd[i][0])

        for j in jList:
            robot = self.scene.robots[j]
            pijx = self.xi.xp - robot.xi.xp
            pijy = self.xi.yp - robot.xi.yp
            pij0 = self.xi.distancepTo(robot.xi)
            pijd0 = self.desired_distance
            tauij0 = (pij0 - pijd0) / pij0
            tauix += tauij0 * pijx
            tauiy += tauij0 * pijy


        # Achieve and keep formation
        # tauix, tauiy = saturate(tauix, tauiy, dxypMax)
        vxp += -K3 * tauix
        vyp += -K3 * tauiy



        ##### transform speed to wheels
        kk = 1
        theta = self.xi.theta
        M11 = kk * math.sin(theta) + math.cos(theta)
        M12 = -kk * math.cos(theta) + math.sin(theta)
        M21 = -kk * math.sin(theta) + math.cos(theta)
        M22 = kk * math.cos(theta) + math.sin(theta)

        v1 = M11 * vxp + M12 * vyp
        v2 = M21 * vxp + M22 * vyp

        vmax = self.control_vmax  # wheel's max linear speed in m/s
        vmin = self.control_vmin # wheel's min linear speed in m/s

        # Find the factor for converting linear speed to angular speed
        if math.fabs(v2) >= math.fabs(v1) and math.fabs(v2) > vmax:
            alpha = vmax / math.fabs(v2)
        elif math.fabs(v2) < math.fabs(v1) and math.fabs(v1) > vmax:
            alpha = vmax / math.fabs(v1)
        else:
            alpha = 1
        v1 = alpha * v1
        v2 = alpha * v2
        # if math.fabs(v1)<vmin:
        #     v1=0
        # if math.fabs(v2)<vmin:
        #     v2=0
        return v1,v2
    def gnn_control(self,omlist,index):
        ####################### NN CONTROLLER ###########################################
        #################################################################################

        # observation, action_1 = self.data.getObservation()
        # if observation is None:
        #    action = np.array([[0, 0]])
        # else:
        action = self.learnedController(omlist, index)
        # action = self.learnedController(observation, self.graph_matrix, action_1[0][0],self.scene.alpha)
        # action = np.array([[0, 0]])
        action = action[0].cpu().detach().numpy()
        v1nn = action[0][0]
        v2nn = action[0][1]
        # smoothing
        self.ctrl1_sm.append(v1nn)
        self.ctrl2_sm.append(v2nn)
        if len(self.ctrl1_sm) < 10:
            v1nn = sum(self.ctrl1_sm) / len(self.ctrl1_sm)
            v2nn = sum(self.ctrl2_sm) / len(self.ctrl2_sm)
        else:
            v1nn = sum(self.ctrl1_sm[len(self.ctrl1_sm) - 10:len(self.ctrl1_sm)]) / 10
            v2nn = sum(self.ctrl2_sm[len(self.ctrl2_sm) - 10:len(self.ctrl2_sm)]) / 10

        # stopping condition
        current_position = (self.xi.xp ** 2 + self.xi.yp ** 2) ** 0.5
        # self.position_hist.append(current_position)
        # hist_len = len(self.position_hist)
        # lcheck = 10
        # if (hist_len > 100):
        #     currhist = self.position_hist[-1 * lcheck:]
        #     if (not self.checkMove(currhist, num=lcheck, thresh=.00001)):
        #         v1nn = 0
        #         v2nn = 0
            # for pos in range(1,len(currhist)):
            #    if(abs(currhist[pos] - currhist[pos - 1]) > .005):
            #        moving = True
            # if(not moving):
            #        print('stopped')
            #        v1nn = 0
            #        v2nn = 0
            # if(abs(self.position_hist[hist_len - 1] - self.position_hist[hist_len-2]) / self.position_hist[hist_len-1] < .0001):
            #    v1nn = 0
            #    v2nn = 0
        ### post-processing ###


        vmax = self.control_vmax  # wheel's max linear speed in m/s
        vmin = self.control_vmin  # wheel's min linear speed in m/s

        # Find the factor for converting linear speed to angular speed
        if math.fabs(v2nn) >= math.fabs(v1nn) and math.fabs(v2nn) > vmax:
            alpha = vmax / math.fabs(v2nn)
        elif math.fabs(v2nn) < math.fabs(v1nn) and math.fabs(v1nn) > vmax:
            alpha = vmax / math.fabs(v1nn)
        else:
            alpha = 1
        v1nn = alpha * v1nn
        v2nn = alpha * v2nn
        # if math.fabs(v1nn) < vmin:
        #     v1nn = 0
        # if math.fabs(v2nn) < vmin:
        #     v2nn = 0
        # Limit maximum acceleration (deprecated)


        self.v1Desirednn = v1nn
        self.v2Desirednn = v2nn

        return v1nn,v2nn
    def control(self,omlist,index,average_distance_gabreil_error,thresh):
        v1_expert,v2_expert=self.expert_control(omlist,index)
        v1_model,v2_model=self.gnn_control(omlist,index)

        # Record data
        if (self.scene.vrepConnected and
            self.scene.SENSOR_TYPE == "VPL16" and
            self.VPL16_counter == 3 and self.recordData == True):
            self.data.add()

        # print('v = ', pow(pow(v1, 2) + pow(v2, 2), 0.5))

        ######## Select either model-based control of NN control ########
        #################################################################
        #print('\n NN output: ', v1nn, v2nn)
        #print('\n Expert output: ', v1, v2)

        ####### TO ADD THE CONTROLLER SELECTION MECHANISM HERE #############
        print("Average distance")
        print(average_distance_gabreil_error)
        thresh=self.nr*thresh

        if self.if_train:
            if not self.use_dagger:
                print("use expert controller")

                v1 = v1_expert
                v2 = v2_expert
                if self.expert_velocity_adjust:
                    print("use expert velocity adjust ")
                    v1 = v1_expert * min(thresh, abs(average_distance_gabreil_error) / self.desired_distance) / thresh
                    v2 = v2_expert * min(thresh, abs(average_distance_gabreil_error) / self.desired_distance) / thresh
            else:
                p = self.p  # can be tweaked
                exp = (self.scene.runNum) // 50
                # exp = (self.scene.runNum-101)//20
                exp = max(0, exp)
                beta = p ** (exp)  # Dagger algorithm paper, page 4
                expert_controller = np.random.binomial(1, beta)
                if expert_controller:
                    print("use expert controller")
                    v1 = v1_expert
                    v2 = v2_expert
                    if self.expert_velocity_adjust:
                        print("use expert velocity adjust ")
                        v1 = v1_expert * min(thresh,abs(average_distance_gabreil_error) / self.desired_distance) / thresh
                        v2 = v2_expert * min(thresh,abs(average_distance_gabreil_error) / self.desired_distance) / thresh
                else:
                    print("use model controller")
                    v1 = v1_model
                    v2 = v2_model
        else:
            if self.expert_only:
                print("use expert controller")

                v1 = v1_expert
                v2 = v2_expert

                if self.expert_velocity_adjust:
                    print("use expert velocity adjust ")
                    v1 = v1_expert * min(thresh,abs(average_distance_gabreil_error) / self.desired_distance) / thresh
                    v2 = v2_expert * min(thresh,abs(average_distance_gabreil_error) / self.desired_distance) / thresh
            else:
                print("use model controller")
                v1 = v1_model
                v2 = v2_model


        # if math.fabs(v1)<self.control_vmin:
        #     v1=0
        # if math.fabs(v2)<self.control_vmin:
        #     v2=0

        self.v1Desired = v1_expert
        self.v2Desired = v2_expert
        if self.scene.vrepConnected:
            # print(v1,v2)
            omega1 = v1 * 10.25
            omega2 = v2 * 10.25
            # omega1 = v1
            # omega2 = v2
            # return angular speeds of the two wheels
            self.nnv1 = omega1
            self.nnv2 = omega2
            self.wheel_velocity_1=v1
            self.wheel_velocity_2=v2
            # return omega1, omega2
            return omega1, omega2
        else:
            # return linear speeds of the two wheels
            self.wheel_velocity_1=v1
            self.wheel_velocity_2=v2
            return v1, v2

    ##### (not yet finish) For update_state
    #### Set one robot's Position and Orientation
    def update_pose(self,stateVector):
        """
        For update_state. Update robot state and return robots pose for scene to execute
        Args:
            stateVector: [x,y,theta] the desire pose of robot
        Returns:
            self.index
            handle:
            position:
            orientation:

        """
        z0 = 0.1587
        if stateVector == None:
            x0 = self.xi.x
            y0 = self.xi.y
            theta0 = self.xi.theta
        elif len(stateVector) == 3:
            x0 = stateVector[0]
            y0 = stateVector[1]
            theta0 = stateVector[2]
            self.xi.x = x0
            self.xi.y = y0
            self.xi.theta = theta0
        else:
            raise Exception('Argument error!')
        position = [x0, y0, z0]
        orientation = [0, 0, theta0]
        handle = self.robotHandle

        return self.index, handle, position, orientation
    #### Set one robot's neighbors
    ##### (not yet finish) For update_state
    def update_neighbors(self,adjmatrix,robot_list):
        """
        Set the neighbors of one robot according to the adjmatrix
        Args:
            adjmatrix: Scene's adjacent matrix
            robot_list: All robots in the scene

        Returns: None

        """
        self.neighbors = []
        self.leader = None
        for j in range(len(robot_list)):
            if adjmatrix[self.index, j] == 0:
                continue
            robot = robot_list[j]  # neighbor
            self.neighbors.append(robot)
    #### (not yet finish)
    def update_state(self,stateVector):
        self.update_pose(stateVector)
        # self.update_neighbors(adjmatrix, robot_list)
    def record_state(self):
        self.pose_list.append([self.xi.x,self.xi.y,self.xi.theta])
        self.velocity_list.append([self.wheel_velocity_1,self.wheel_velocity_2])
#### Get one robot's Position, Orientation and Lidar reading
    def get_sensor_data(self,pos,ori,vel,omega,velodyne_points):
        """
        Get pose from simulator and sensor
        Args:
            pos: robot position from simulator
            ori: robot orientation from simulator
            vel: robot linear velocity from simulator
            omega: robot angular velocity from simulator
            velodyne_points: sensor data

        Returns: None

        """
        ##### Get absolute pose from simulator
        self.xi.x = pos[0]
        self.xi.y = pos[1]
        self.xi.alpha = ori[0]
        self.xi.beta = ori[1]
        self.xi.theta = ori[2]
        sgn = np.sign(np.dot(np.asarray(vel[0:2]),
                             np.asarray([math.cos(self.xi.theta),
                                         math.sin(self.xi.theta)])))
        self.vActual = sgn * (vel[0]**2 + vel[1]**2)**0.5
        self.omegaActual = omega[2]
        ##### Get laser/vision sensor data
        # Parse data
        if not velodyne_points==None:
            # res = velodyne_points[0]
            if 'VPL16_counter' not in self.__dict__:
                self.VPL16_counter = 0
            # reset the counter every fourth time
            if self.VPL16_counter == 4:
                self.VPL16_counter = 0
            if self.VPL16_counter == 0:
                # Reset point cloud
                self.pointCloud.clearData()
            self.pointCloud.addRawData(velodyne_points[2]) # will rotate here

            if self.VPL16_counter == 3:
                self.pointCloud.crop()
                self.pointCloud.updateOccupancyMap() # option 1
            self.VPL16_counter += 1
            # print("counter")
            # print(self.VPL16_counter)
            # add for gnn
            ###### (not yet finish)
            self.graph_matrix = self.scene.readADjMatrix(MaxRange=2)
    def getV1V2(self):
        v1 = self.vActual + self.omegaActual * self.l / 2
        v2 = self.vActual - self.omegaActual * self.l / 2
        return np.array([[v1, v2]])
##### might move to scene
##### pending decide
    # def propagateDesired(self):
    #     """
    #     Update robot desire pose
    #     Returns:None
    #
    #     """
    #     self.xid.theta = self.scene.xid.vRefAng

##### For simulate
    def precompute(self,adjmatrix,robot_list):
        self.xi.transform()
        self.xid.transform()
        self.update_neighbors(adjmatrix,robot_list)
#### Set the linear velocity of 2 wheels

    def get_control(self,omlist,i,average_distance_gabreil_error,thresh):
        if self.scene.vrepConnected == False:
            self.xi.propagate(self.control)
        else:
            omega1, omega2 = self.control(omlist,i,average_distance_gabreil_error,thresh)
            return omega1,omega2
    ##### For simulate in scene
    def getDataObs(self):
        observation, action_1 = self.data.getObservation(-12)
        return observation, self.graph_matrix, action_1[0][0],self.scene.alpha
    def save_trace(self,path):
        pose_array=np.array(self.pose_list)
        velocity_array=np.array(self.velocity_list)
        pose_path = path + "pose_array_robot_" + str(self.index) + ".npy"
        velocity_path = path + "velocity_array_robot_" + str(self.index) + ".npy"
        np.save(pose_path,pose_array)
        np.save(velocity_path,velocity_array)
        print("Pose array of robot "+str(self.index)+" saved at "+pose_path)
        print("Velocity array of robot " + str(self.index) + " saved at " + velocity_path)

    def setPosition(self, stateVector = None):
        # stateVector = [x, y, theta]

        z0 = 0.1587
        # if stateVector == None:
        #     x0 = self.xi.x
        #     y0 = self.xi.y
        #     theta0 = self.xi.theta
        if len(stateVector) == 3:
            x0 = stateVector[0]
            y0 = stateVector[1]
            theta0 = stateVector[2]
            self.xi.x = x0
            self.xi.y = y0
            self.xi.theta = theta0
        else:
            raise Exception('Argument error!')
        if self.scene.vrepConnected == False:
            return
        vrep.simxSetObjectPosition(self.scene.clientID, self.robotHandle, -1,
                [x0, y0, z0], vrep.simx_opmode_oneshot)
        vrep.simxSetObjectOrientation(self.scene.clientID, self.robotHandle, -1,
                [0, 0, theta0], vrep.simx_opmode_oneshot)
        message = "Robot #" + str(self.index) + "'s pose is set to "
        message += "[{0:.3f}, {1:.3f}, {2:.3f}]".format(x0, y0, theta0)
        self.scene.log(message)




