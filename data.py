# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 11:49:17 2018

@author: Zhuo Chen
"""
import numpy as np
import os
import queue
import math

class Data():
    def __init__(self, robot):
        self.q = queue.Queue()
        self.robot = robot
        dynamics = 18
        pc = self.robot.pointCloud
        self.d = dict() # Will become None after the scene is saved
        self.d['epi_starts'] = np.array([], dtype = np.bool)
        self.d['observations'] = np.zeros((0, pc.hPix * pc.wPix), dtype = np.int8)
        self.d['observations1'] = np.zeros((0, pc.lenScanVector), dtype = np.float32)
        self.d['obs2'] = np.zeros((0,robot.nr + 5),dtype = np.float32)
        self.d['actions'] = np.zeros((0, 2), dtype = np.float32)
        # add communication graph for gnn
        #self.d['graph'] = np.zeros((0, 9), dtype = np.float32) # 9: n by n where n is the # of robots
        self.d['graph'] = np.zeros((0,robot.nr**2), dtype=np.float32)
    def getObservation(self):
        obs0 = self.robot.pointCloud.getObservation()

        peer = self.robot
        psi = peer.xid.theta - peer.xi.theta
        if psi > math.pi:
            psi -= 2 * math.pi
        elif psi < -math.pi:
            psi += 2 * math.pi
        dpbar = (peer.scene.xid.dpbarx**2 + peer.scene.xid.dpbary**2)**0.5
        dstars = []
        for neighbor in self.robot.neighbors:
            dstars.append(peer.xid.distancepTo(neighbor.xid))
#                state = np.array([[dpbar, psi,
#                                   peer.xi.x, peer.xi.y, peer.xi.theta]])
        #state = np.array([[dpbar, psi, peer.scene.alpha] + dstars])
        state = np.array([[psi, peer.scene.alpha]])
        # print("state")
        # print(state)
        ret = (obs0, state)

        return ret

    def add(self):
        # Add data corresponding to a particular point in time
        # This function can only run after the leader state is updated
        # This function can only run before self.robot is desctructed
        # This function can not run after scene has been saved as a pickle file
        observation, observation2 = self.getObservation()
        if observation is None:
            return

        if len(self.d['epi_starts']) == 0:
            self.d['epi_starts'] = np.append(self.d['epi_starts'], True)
        else:
            self.d['epi_starts'] = np.append(self.d['epi_starts'], False)

        self.d['observations'] = np.append(self.d['observations'], observation, axis = 0) # option 1

        self.d['observations1'] = np.append(self.d['observations1'],
                              self.robot.pointCloud.scanVector, axis = 0) # option 2
        peer = self.robot
        psi = peer.xid.theta - peer.xi.theta
        if psi > math.pi:
            psi -= 2 * math.pi
        elif psi < -math.pi:
            psi += 2 * math.pi
        dpbar = (peer.scene.xid.dpbarx**2 + peer.scene.xid.dpbary**2)**0.5
        dstars = []
        for neighbor in self.robot.neighbors:
            dstars.append(peer.xid.distancepTo(neighbor.xid))
        #print(dstars)
        obs2Data = [[dpbar, psi, peer.scene.alpha,
                     peer.xi.x, peer.xi.y, peer.xi.theta]
                    + dstars] # mode = -12
            #print("Robot", self.robot.index, ", psi: ", psi)
        self.d['obs2'] = np.append(self.d['obs2'], obs2Data, axis = 0)
        self.d['actions'] = np.append(self.d['actions'],
                  [[self.robot.v1Desired, self.robot.v2Desired]], axis = 0)
        # add graph for gnn
        self.d['graph'] = np.append(self.d['graph'],
                  self.robot.graph_matrix, axis = 0)

    def append(self, data2):
        # Append data collected in a run
        for key in self.d:
            self.d[key] =  np.append(self.d[key], data2.d[key], axis = 0)

    def store(self):
        i = self.robot.index
        directory = 'data'
        if not os.path.exists(directory):
            os.makedirs(directory)
        count = 0
        for filename in os.listdir(directory):
            if filename[0:4] != "data" or filename[-4:] != ".npz":
                continue
            name, _ = os.path.splitext(filename)
            n = int(name[4:7])
            if count < n:
                count = n
        count += 1
        path = os.path.join(directory, 'data' + str(count).zfill(3) + '_' + str(i))
        np.savez(path, **(self.d))
        message = "Training data of length {0:d} saved to " + path + ".npz"
        message = message.format(len(self.d['epi_starts']))
        self.robot.scene.log(message)


