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
        self.mode = -12
        self.q = queue.Queue()
        self.robot = robot
        dynamics = 18
        pc = self.robot.pointCloud
        self.d = dict() # Will become None after the scene is saved
        self.d['epi_starts'] = np.array([], dtype = np.bool)
        if self.mode < 0:
            self.d['observations'] = np.zeros((0, pc.hPix * pc.wPix), dtype = np.int8)
        elif self.mode > 0:
            self.d['observations'] = np.zeros((0, pc.hPix * pc.wPix * 2), dtype = np.int8)
            self.d['observations2'] = np.zeros((0, 2), dtype = np.float32)
        self.d['observations1'] = np.zeros((0, pc.lenScanVector), dtype = np.float32)
        print('Dynamics:',dynamics)
            #self.d['obs2'] = np.zeros((0, 6), dtype = np.float32)
        self.d['obs2'] = np.zeros((0,robot.nr + 5),dtype = np.float32)
        self.d['actions'] = np.zeros((0, 2), dtype = np.float32)
        # add communication graph for gnn
        #self.d['graph'] = np.zeros((0, 9), dtype = np.float32) # 9: n by n where n is the # of robots
        self.d['graph'] = np.zeros((0,robot.nr**2), dtype=np.float32)
    def getObservation(self, mode):
        # This function can not run after scene has been saved as a pickle file
        if mode == 0:
            obs0 = self.robot.pointCloud.getObservation()
            act0 = self.robot.getV1V2()
            ret = (obs0, act0)
        elif mode > 0:
            obs0 = self.robot.pointCloud.getObservation()
            xi0 = self.robot.xi
            #print(self.q.qsize())
            if self.q.qsize() == mode:
                xi1 = self.q.get()
                obs = np.concatenate(xi1, axis = 1)
                ret = (obs, None)
            else:
                ret = (None, None)
            self.q.put(xi0)
        elif mode < 0:
            obs0 = self.robot.pointCloud.getObservation()
            if mode == -1: # 1x1 Leader's actual speed
                vLeader = self.robot.leader.getV1V2()
                state = np.array([[0.5*(vLeader[0, 0] + vLeader[0, 1])]])
            elif mode == -2: # 1x1 Follower's reference speed
                followerXid = self.robot.xid
                state = np.array([[(followerXid.vx**2 + followerXid.vy**2)**0.5]])
            elif mode == -3: # 1x2 Follower's reference speed
                followerXid = self.robot.xid
                state = np.array([[followerXid.vx, followerXid.vy]])
            elif mode == -4: # 1x2 Velocities of the leader's two wheels
                state = self.robot.leader.getV1V2()
            elif mode == -10: # Peer's state
                peer = self.robot
                phii = math.atan2(peer.xid.y - peer.xi.y, peer.xid.x - peer.xi.x)
                rhoi = ((peer.xid.x - peer.xi.x)**2 + (peer.xid.y - peer.xi.y)**2) ** 0.5
                thetai = peer.xi.theta
                psi = phii - thetai
                if psi > math.pi:
                    psi -= 2 * math.pi
                elif psi < -math.pi:
                    psi += 2 * math.pi
                state = np.array([[peer.xid.vRef, rhoi, psi]])
            elif mode == -11: # Peer's state
                peer = self.robot
                state = np.array([[peer.xi.x, peer.xi.y, peer.xid.x, peer.xid.y]])
            elif mode == -12: # Peer's state
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
        observation, observation2 = self.getObservation(self.mode)
        if observation is None:
            return

        if len(self.d['epi_starts']) == 0:
            self.d['epi_starts'] = np.append(self.d['epi_starts'], True)
        else:
            self.d['epi_starts'] = np.append(self.d['epi_starts'], False)

        self.d['observations'] = np.append(self.d['observations'], observation, axis = 0) # option 1
        if self.mode >= 0:
            self.d['observations2'] = np.append(self.d['observations2'], observation2, axis = 0)

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


