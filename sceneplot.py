# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 17:22:47 2018

@author: cz
"""
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import os



class ScenePlot():
    def __init__(self, scene = None, saveEnabled = True):
        self.sc = scene
        self.TYPE_TIME_SEPARATION_ERROR = 2
        self.TYPE_TIME_BEARING_ERROR = 3
        self.TYPE_TIME_ACTIONS = 6
        self.getRobotColor = self.sc.getRobotColor

        self.saveEnabled = saveEnabled
        if self.saveEnabled == True:
            directory = 'fig'
            if not os.path.exists(directory):
                os.makedirs(directory)
            self.directory = directory

            count = 0
            for filename in os.listdir(directory):
                name, _ = os.path.splitext(filename)
                n = int(name)
                if count < n:
                    count = n
            count = count + 1
            self.directory = os.path.join(self.directory, str(count).zfill(3))
            os.makedirs(self.directory)




    def plot(self, type = 0, tf = 0,expert=False):
        # type 0: (t, x_i - x_id)
        # type 1: (t, y_i - y_id)
        # type 2: (t, e_i - e_id) (formation error)
        # type 3: (x_i, y_i)
        # type 4: (t, v1)
        # If this is the first time this type of plot is drawn, initialize
        if type not in self.sc.ydict.keys():
            self.sc.ydict[type] = dict()
            self.sc.ydict2[type] = dict() # for type 3 figure only
            self.sc.ploted[type] = False
            if type == 0 or type == 1:
                for i in range(len(self.sc.robots)):
                    self.sc.ydict[type][i] = []

        if self.sc.ploted[type] and type != 6:
            pass#return # This type of plot is completed
        if type == 0:
            if not self.sc.ploted[type]:
                for i in range(len(self.sc.robots)):
                    x = self.sc.robots[i].xi.x - self.sc.robots[i].xid.x
                    self.sc.ydict[type][i].append(x)
            if self.sc.t > tf:
                plt.figure(type)
                for i in range(len(self.sc.robots)):
                    try:
                        plt.plot(self.sc.ts, self.sc.ydict[type][i], '-')
                    except:
                        print('type: ', type)
                        raise
                plt.xlabel('t (s)')
                plt.ylabel('x_i - x_di (m)')

        elif type == 1:
            if not self.sc.ploted[type]:
                for i in range(len(self.sc.robots)):
                    x = self.sc.robots[i].xi.y - self.sc.robots[i].xid.y
                    self.sc.ydict[type][i].append(x)
            if self.sc.t > tf:
                plt.figure(type)
                for i in range(len(self.sc.robots)):
                    plt.plot(self.sc.ts, self.sc.ydict[type][i], '-')
                plt.xlabel('t (s)')
                plt.ylabel('y_i - y_di (m)')

        elif type == 2: # Formation Error type 2
            if not self.sc.ploted[type]:
                k = 0
                for i in range(0, len(self.sc.robots)):
                    xi = self.sc.robots[i].xi.x
                    yi = self.sc.robots[i].xi.y
                    xid = self.sc.robots[i].xid.x
                    yid = self.sc.robots[i].xid.y

                    if self.sc.dynamics == 5:
                        #jList = range(0, 1)
                        jList = range(0, len(self.sc.robots))
                    elif self.sc.dynamics >= 16 and self.sc.dynamics <= 17:
                        jList = range(0, len(self.sc.robots))
                    elif self.sc.dynamics == 18:
                        #lsd = self.sc.robots[i].listSortedDistance
                        #jList = [lsd[0][0],lsd[1][0]]
                        jList = range(0, len(self.sc.robots))
                    else:
                        raise Exception("Undefined robot dynamics for data recording", self.sc.dynamics)
                    for j in jList:
                        if self.sc.adjMatrix[i, j] == 0:
                            continue
                        if i > j and self.sc.adjMatrix[j, i] != 0:
                            continue
                        xj = self.sc.robots[j].xi.x
                        yj = self.sc.robots[j].xi.y
                        xjd = self.sc.robots[j].xid.x
                        yjd = self.sc.robots[j].xid.y

                        eji = np.array([xi - xj, yi - yj])
                        if self.sc.dynamics == 5:
                            ejid = np.array([xid - xjd, yid - yjd])
                        elif self.sc.dynamics == 17:
                            ejid = np.array([xid - xjd, yid - yjd])
                        else:
                            ejid = self.sc.alpha
                        error = np.linalg.norm(eji) #- np.linalg.norm(ejid)

                        # If this is the first time this type of plot is drawn
                        if k not in self.sc.ydict[type].keys():
                            self.sc.ydict[type][k] = []
                            self.sc.ydict2[type][k] = (i, j) # for legend
                        self.sc.ydict[type][k].append(error)

                        k += 1

            if self.sc.t > tf:
                errors = self.sc.ydict[type]
                legends = self.sc.ydict2[type]
                curves = []
                plt.figure(type)
                for k in range(0, len(self.sc.ydict[type])):
                    curve, = plt.plot(self.sc.ts, errors[k], '-', label = str(legends[k]))
                    curves.append(curve)
                if int(matplotlib.__version__[0]) == 2:
                    plt.legend(handles = curves)
                plt.xlabel('t (s)')
                plt.ylabel('Formation Separation Error (m)')

        elif type == 21: # Formation orientation error
            if not self.sc.ploted[type]:
                k = 0
                for i in range(1, len(self.sc.robots)):
                    xi = self.sc.robots[i].xi.x
                    yi = self.sc.robots[i].xi.y
                    xid = self.sc.robots[i].xid.x
                    yid = self.sc.robots[i].xid.y

                    if self.sc.dynamics == 13:
                        j2 = 1
                    elif self.sc.dynamics == 14 or self.sc.dynamics == 16 or self.sc.dynamics == 17:
                        j2 = i
                    for j in range(0, j2):
                        if self.sc.adjMatrix[i, j] == 0:
                            continue
                        xj = self.sc.robots[j].xi.x
                        yj = self.sc.robots[j].xi.y
                        xjd = self.sc.robots[j].xid.x
                        yjd = self.sc.robots[j].xid.y

                        eji = np.array([xi - xj, yi - yj])
                        ejid = np.array([xid - xjd, yid - yjd])
                        angle = math.acos(np.inner(eji, ejid) /
                             np.linalg.norm(eji) / np.linalg.norm(ejid))

                        # If this is the first time this type of plot is drawn
                        if k not in self.sc.ydict[type].keys():
                            self.sc.ydict[type][k] = []
                            self.sc.ydict2[type][k] = (i, j) # for legend
                        self.sc.ydict[type][k].append(angle)

                        k += 1

            if self.sc.t > tf:
                errors = self.sc.ydict[type]
                legends = self.sc.ydict2[type]
                curves = []
                plt.figure(type)
                for k in range(0, len(self.sc.ydict[type])):
                    curve, = plt.plot(self.sc.ts, errors[k], '-', label = str(legends[k]))
                    curves.append(curve)
                if int(matplotlib.__version__[0]) == 2:
                    plt.legend(handles = curves)
                plt.xlabel('t (s)')
                plt.ylabel('Formation Orientation Error (rad)')

        elif type == 22: # Distance from goal
            if not self.sc.ploted[type]:
                for i in range(0, len(self.sc.robots)):
                    # If this is the first time this type of plot is drawn
                    if i not in self.sc.ydict[type].keys():
                        self.sc.ydict[type][i] = []
                        # print(self.sc.ydict[type].keys())
                        # print('i = ', i, 'j = ', j)
                    xi = self.sc.robots[i].xi.x
                    yi = self.sc.robots[i].xi.y
                    xid = self.sc.robots[i].xid.x
                    yid = self.sc.robots[i].xid.y

                    distance = ((xid - xi)**2 + (yid - yi)**2)**0.5

                    self.sc.ydict[type][i].append(distance)
                    #print(self.sc.ydict[type][i])
            if self.sc.t > tf:
                errors = self.sc.ydict[type]
                plt.figure(type)
                for i in range(0, len(self.sc.robots)):
                    plt.plot(self.sc.ts, errors[i], '-')
                plt.xlabel('t (s)')
                plt.ylabel('Distance from goal (m)')

        elif type == 23: # Ceter distance from goal
            if not self.sc.ploted[type]:
                # If this is the first time this type of plot is drawn
                if 0 not in self.sc.ydict[type].keys():
                    self.sc.ydict[type][0] = []
                xbar = 0
                ybar = 0
                for i in range(0, len(self.sc.robots)):
                    xbar += self.sc.robots[i].xi.x
                    ybar += self.sc.robots[i].xi.y
                xbar /= len(self.sc.robots)
                ybar /= len(self.sc.robots)
                xd = self.sc.xid.x
                yd = self.sc.xid.y
                distance = ((xbar - xd)**2 + (ybar - yd)**2)**0.5
                self.sc.ydict[type][0].append(distance)
                #print(self.sc.ydict[type][0])

            if self.sc.t > tf:
                error = self.sc.ydict[type][0]
                plt.figure(type)
                plt.plot(self.sc.ts, error, '-', color = (0,0,0))
                plt.xlabel('t (s)')
                plt.ylabel('Center distance from goal (m)')

        elif type == 24: # Formation Error type 24: epsilon
            if len(self.sc.robots) != 3:
                raise Exception("Plot type 24 is valid only when the number of robots is 3.")
            if self.sc.dynamics != 5:
                raise Exception("Plot type 24 is valid only for dynamics 13.")
            xi = 3 * [None]
            xid = 3 * [None]
            if not self.sc.ploted[type]:
                for i in range(0, 3):
                    xi[i] = self.sc.robots[i].xi
                    xid[i] = self.sc.robots[i].xid

                for i in range(0, 3):
                    error = 0
                    j1 = None
                    for j in range(0, 3):
                        if i == j:
                            continue
                        if j1 is None:
                            j1 = j
                        else:
                            j2 = j
                        xj = self.sc.robots[j].xi.x
                        yj = self.sc.robots[j].xi.y
                        xjd = self.sc.robots[j].xid.x
                        yjd = self.sc.robots[j].xid.y

                    errorij1 = xi[i].distancepTo(xi[j1]) - xid[i].distancepTo(xid[j1])
                    errorij2 = xi[i].distancepTo(xi[j2]) - xid[i].distancepTo(xid[j2])
                    errorj1j2 = xi[j1].distancepTo(xi[j2]) - xid[j1].distancepTo(xid[j2])
                    ejid = xid[i].distancepTo(xid[j])
                    epsilon = errorij1 + errorij2 - errorj1j2

                    # If this is the first time this type of plot is drawn
                    if i not in self.sc.ydict[type].keys():
                        self.sc.ydict[type][i] = []
                        self.sc.ydict2[type][i] = i # for legend
                    self.sc.ydict[type][i].append(epsilon)
            if self.sc.t > tf:
                errors = self.sc.ydict[type]
                legends = self.sc.ydict2[type]
                curves = []
                plt.figure(type)
                for k in range(0, len(self.sc.ydict[type])):
                    curve, = plt.plot(self.sc.ts, errors[k], '-',
                                      label = str(legends[k]))
                    curves.append(curve)
                if int(matplotlib.__version__[0]) == 2:
                    plt.legend(handles = curves)
                plt.xlabel('t (s)')
                plt.ylabel('Formation Separation Error (m)')

        elif type == 3: # Formation Error type 3
            if not self.sc.ploted[type]:
                for i in range(1, len(self.sc.robots)):
                    # If this is the first time this type of plot is drawn
                    if i not in self.sc.ydict[type].keys():
                        self.sc.ydict[type][i] = []
                        # print(self.sc.ydict[type].keys())
                        # print('i = ', i, 'j = ', j)
                    xi = self.sc.robots[i].xi.x
                    yi = self.sc.robots[i].xi.y
                    xid = self.sc.robots[i].xid.x
                    yid = self.sc.robots[i].xid.y
                    j = 0
                    xj = self.sc.robots[j].xi.x
                    yj = self.sc.robots[j].xi.y
                    xjd = self.sc.robots[j].xid.x
                    yjd = self.sc.robots[j].xid.y

                    phi = math.atan2(yi - yj, xi - xj)
                    phid = math.atan2(yid - yjd, xid - xjd)
                    error = phi - phid
                    while error > math.pi:
                        error -= 2 * math.pi
                    while error <= -math.pi:
                        error += 2 * math.pi

                    self.sc.ydict[type][i].append(error)
                    #print(self.sc.ydict[type][i])
            if self.sc.t > tf:
                errors = self.sc.ydict[type]
                plt.figure(type)
                for i in range(1, len(self.sc.robots)):
                    plt.plot(self.sc.ts, errors[i], '-')
                plt.xlabel('t (s)')
                plt.ylabel('Bearing Error (rad)')

        elif type == 4:
            # Show formation
            if not self.sc.ploted[type]:
                # record individual trajectories
                for i in range(len(self.sc.robots)):
                    x = self.sc.robots[i].xi.x
                    y = self.sc.robots[i].xi.y
                    if i not in self.sc.ydict[type].keys():
                        self.sc.ydict2[type][i] = np.array([[x, y]])
                    else:
                        self.sc.ydict2[type][i] = np.append(self.sc.ydict2[type][i],
                                   [[x, y]], axis = 0)

                # print('time: ', (self.sc.t + 1e-5) % 1)
                if (self.sc.t + 1e-5) % 3 < 2e-5:
                    # print("recording")
                    self.sc.tss.append(self.sc.t)
                    for i in range(len(self.sc.robots)):
                        x = self.sc.robots[i].xi.x
                        y = self.sc.robots[i].xi.y
                        theta = self.sc.robots[i].xi.theta*180/math.pi - 90 # convert to deg
                        if len(self.sc.tss) == 1:
                            self.sc.ydict[type][i] = np.array([[x, y, theta]])
                            if i == 0:
                                self.sc.centerTrajS = np.array([[x, y]])
                            else:
                                self.sc.centerTrajS += np.array([[x, y]])
                        else:
                            self.sc.ydict[type][i] = np.append(self.sc.ydict[type][i],
                                      [[x, y, theta]], axis = 0)
                            if i == 0:
                                self.sc.centerTrajS = np.append(self.sc.centerTrajS, [[x, y]], axis = 0)
                            else:
                                self.sc.centerTrajS[-1, :] += np.array([x, y])
                        #print(self.sc.centerTrajS)
                    self.sc.centerTrajS[-1, :] /= len(self.sc.robots)

            # Show Figure
            if self.sc.t > tf:
                plt.figure(type)
                curves = []
                if self.sc.dynamics == 5:
                    style = 3
                else:
                    style = 0
                for i in range(len(self.sc.robots)):
                    c = self.getRobotColor(i)
                    curve, = plt.plot(self.sc.ydict2[type][i][:, 0],
                             self.sc.ydict2[type][i][:, 1],
                             '-', color = c, label = str(i))
                    curves.append(curve)
                    for k in range(len(self.sc.tss)):
                        plt.plot(self.sc.ydict[type][i][k, 0],
                                 self.sc.ydict[type][i][k, 1],
                                 marker=(3, style, self.sc.ydict[type][i][k, 2]),
                                 markersize=20, linestyle='None',
                                 color=c, fillstyle="none")
                if int(matplotlib.__version__[0]) == 2:
                    plt.legend(handles = curves)
                # for k in range(len(self.sc.tss)):
                #     for i in range(len(self.sc.robots)):
                #         x1 = self.sc.ydict[type][i][k, 0]
                #         y1 = self.sc.ydict[type][i][k, 1]
                #         for j in range(0, len(self.sc.robots)):
                #             if self.sc.adjMatrix[i, j] == 0:
                #                 continue
                #             if i > j and self.sc.adjMatrix[j, i] != 0:
                #                 continue
                #             x2 = self.sc.ydict[type][j][k, 0]
                #             y2 = self.sc.ydict[type][j][k, 1]
                #             plt.plot([x1, x2], [y1, y2], ':', color = (0, 0, 0))



                # # Plot center trajectory
                # for k in range(len(self.sc.tss)):
                #     plt.plot(self.sc.centerTrajS[k, 0],
                #              self.sc.centerTrajS[k, 1],
                #              'o',
                #              markersize=10, linestyle='None',
                #              color = (0, 0, 0))

                # plt.plot(self.sc.centerTraj[:, 0],
                #          self.sc.centerTraj[:, 1],
                #          '-',
                #          color = (0, 0, 0))

                plt.xlabel('x (m)')
                plt.ylabel('y (m)')
                plt.axes().set_aspect('equal', 'datalim')


        elif type == 5:
            # Show speed
            if not self.sc.ploted[type]:
                for i in range(len(self.sc.robots)):
                    vDesired = (self.sc.robots[i].v1Desired + self.sc.robots[i].v2Desired)/2
                    if self.sc.vrepConnected ==  True:
                        vActual = self.sc.robots[i].vActual
                    else:
                        vActual = vDesired
                    if i not in self.sc.ydict[type].keys():
                        self.sc.ydict[type][i] = []
                        self.sc.ydict2[type][i] = []
                    self.sc.ydict[type][i].append(vActual)
                    self.sc.ydict2[type][i].append(vDesired)
            if self.sc.t > tf:
                plt.figure(type)
                for i in range(len(self.sc.robots)):
                    c = self.getRobotColor(i)
                    curve1, = plt.plot(self.sc.ts, self.sc.ydict[type][i], '-',
                                      color = c, label = 'Actual')
                    curve2, = plt.plot(self.sc.ts, self.sc.ydict2[type][i], '--',
                                      color = c, label = 'Desired')
                if int(matplotlib.__version__[0]) == 2:
                    plt.legend(handles = [curve1, curve2])
                plt.xlabel('t (s)')
                plt.ylabel('v (m/s)')

        elif type == 6:
            # Show action
            matplotlib.rcParams.update(matplotlib.rcParamsDefault)
            plt.rcParams.update({'font.size': 24})
            plist = [[] for i in range(len(self.sc.robots))]
            if self.sc.dynamics == 13:
                j1 = 1
            elif self.sc.dynamics == 14 or (self.sc.dynamics >= 16 and self.sc.dynamics <= 18):
                j1 = 0
            else:
                j1 = 0
            if not self.sc.ploted[type]:
                for i in range(len(self.sc.robots)):
                    if(expert):
                        vDesired1 = self.sc.robots[i].v1Desired
                        vDesired2 = self.sc.robots[i].v2Desired
                    else:
                        vDesired1 = self.sc.robots[i].v1Desirednn
                        vDesired2 = self.sc.robots[i].v2Desirednn
                    plist[i].append([vDesired1,vDesired2])
                    #vDesired1 /= 5
                    #vDesired2 /= 5
                    if i not in self.sc.ydict[type].keys():
                        self.sc.ydict[type][i] = []
                        self.sc.ydict2[type][i] = []
                    self.sc.ydict[type][i].append(vDesired1)
                    self.sc.ydict2[type][i].append(vDesired2)
            if self.sc.t > tf:
                plt.figure(type,figsize=(11,6))
                for i in range(j1, len(self.sc.robots)): # Show only the follower
                #for i in range(len(self.sc.robots)): # Show both the follower and the leader
                    c = self.getRobotColor(i)
                    if self.sc.dynamics  == 5:
                        curve1Label = 'Action in x-axis'
                        curve2Label = 'Action in y-axis'
                    else:
                        curve1Label = 'Robot ' + str(i+1) + ' Left Wheel Velocity'
                        curve2Label = 'Robot ' + str(i+1) + ' Right Wheel Velocity'
                    curve1, = plt.plot(self.sc.ts, self.sc.ydict[type][i], ':',
                                      color = c, label = curve1Label)
                    curve2, = plt.plot(self.sc.ts, self.sc.ydict2[type][i], '--',
                                      color = c, label = curve2Label)
                #if int(matplotlib.__version__[0]) == 2:
                plt.legend(prop={'size':8},loc='upper right')
                plt.xlabel('Time (s)',fontsize=20)
                plt.ylabel('Robot Actions (m/s)',fontsize=20)
                plt.tight_layout()
                plist = np.array(plist)
                np.save('controllist_'+str(len(self.sc.robots))+'.npy',plist)

        elif type == 7:
            # Show angular velocity
            if not self.sc.ploted[type]:
                for i in range(len(self.sc.robots)):
                    omegaDesired = (self.sc.robots[i].v2Desired -
                                    self.sc.robots[i].v1Desired) / self.sc.robots[i].l
                    if self.sc.vrepConnected ==  True:
                        omegaActual = self.sc.robots[i].omegaActual
                    else:
                        omegaActual = omegaDesired
                    if i not in self.sc.ydict[type].keys():
                        self.sc.ydict[type][i] = []
                        self.sc.ydict2[type][i] = []
                    self.sc.ydict[type][i].append(omegaActual)
                    self.sc.ydict2[type][i].append(omegaDesired)
            if self.sc.t > tf:
                plt.figure(type)
                for i in range(len(self.sc.robots)):
                    c = self.getRobotColor(i)
                    curve1, = plt.plot(self.sc.ts, self.sc.ydict[type][i], '-',
                                      color = c, label = 'Actual')
                    curve2, = plt.plot(self.sc.ts, self.sc.ydict2[type][i], '--',
                                      color = c, label = 'Desired')
                if int(matplotlib.__version__[0]) == 2:
                    plt.legend(handles = [curve1, curve2])
                plt.xlabel('t (s)')
                plt.ylabel('omega (rad/s)')

        elif type == 8:
            # Show Euler angles
            if not self.sc.ploted[type]:
                if self.sc.vrepConnected ==  False:
                    return
                for i in range(len(self.sc.robots)):
                    alpha = self.sc.robots[i].xi.alpha / math.pi * 180
                    beta = self.sc.robots[i].xi.beta / math.pi * 180
                    if i not in self.sc.ydict[type].keys():
                        self.sc.ydict[type][i] = []
                        self.sc.ydict2[type][i] = []
                    self.sc.ydict[type][i].append(alpha)
                    self.sc.ydict2[type][i].append(beta)
            if self.sc.t > tf:
                if len(self.sc.ts) != len(self.sc.ydict[type][0]):
                    return
                plt.figure(type)
                for i in range(len(self.sc.robots)):
                    c = self.getRobotColor(i)
                    curve1, = plt.plot(self.sc.ts, self.sc.ydict[type][i], '-',
                                      color = c, label = 'alpha')
                    curve2, = plt.plot(self.sc.ts, self.sc.ydict2[type][i], '--',
                                      color = c, label = 'beta')
                if int(matplotlib.__version__[0]) == 2:
                    plt.legend(handles = [curve1, curve2])
                plt.xlabel('t (s)')
                plt.ylabel('angles (deg)')

        elif type == 9:
            # Show observation1
            for i in range(len(self.sc.robots)):
                plt.figure(type)
                c = self.getRobotColor(i)
                pc = self.sc.robots[i].pointCloud
                dist = pc.scanVector
                angle = pc.scanAngle
                for j in range(pc.lenScanVector):
                    x = -dist[0, j] * math.cos(angle[j])
                    y = -dist[0, j] * math.sin(angle[j])
                    plt.plot([0, x], [0, y], '-', color = c)
                plt.xlabel('x (m)')
                plt.ylabel('y (m)')
                plt.axes().set_aspect('equal')
                #plt.show()
            self.sc.ploted[type] = True
        else:
            raise Exception("Undefined type of plot.")

        if self.sc.t > tf:
            plt.grid(True)
            if self.saveEnabled == True:
                # Save plot as eps file
                path = os.path.join(self.directory, 'fig' + str(type).zfill(2) + '.png')
                plt.savefig(path)#, dpi=1000)
                message = "Plot saved to " + str(path)
                self.sc.log(message)
                print(message)
                #plt.show()
                plt.close('all')
            else:
                pass
                #plt.show()
            self.sc.ploted[type] = True









