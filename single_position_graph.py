import numpy as np
import matplotlib.pyplot as plt

def gabriel(i,j,ri,rj,l):

    connected = True
    for k in range(len(l)):
        if i == k or j == k:
            continue
        di = ri - l[k]
        dj = rj - l[k]
        c = np.dot(di,dj)/(np.linalg.norm(di)*np.linalg.norm(dj))
        angle = np.arccos(c) * 180 / np.pi
        #d = (ri + rj) / 2
        #r = (np.linalg.norm(ri - d) + np.linalg.norm(rj - d))/2
        if(angle > 89 and i != k  and j != k):
            connected = False
        #if(np.linalg.norm(d - l[k]) <= r):
        #    connected = False

    return connected

#n = 7
#ns = [4,5,6,7,8,9,12]
n = 12
ns = [3,4,5,8,10,12]
#exp = np.load('positionList_single_'+str(n)+'.npy')[0]
alldata = np.load('positionList_expert_'+str(n)+'_longer.npy')
exp = alldata[1]
pos = np.zeros((n,exp.shape[0]//n,2))
for i in range(n):
    check = exp[i::n]
    pos[i] = check

allpos = np.zeros((alldata.shape[0],n,exp.shape[0]//n,2))
for i in range(alldata.shape[0]):
    curr = alldata[i]
    for j in range(n):
        check = curr[j::n]
        allpos[i][j] = check
alldata = allpos
# paths
endpos = pos[:,0,:]
xlim_min = min(endpos[:,0])
xlim_max = max(endpos[:,0])
ylim_min = min(endpos[:,1])
ylim_max = max(endpos[:,1])
plt.figure()
for i in range(pos.shape[0]):
    plt.plot(pos[i,:,0],pos[i,:,1],label=str(i))
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Positions of ' + str(n) + ' robots')
plt.legend()
plt.xlim([xlim_min-1,xlim_max+1])
plt.ylim([ylim_min-1,ylim_max+1])
plt.show()

# end positions
plt.figure()
plt.scatter(pos[:,-1,0],pos[:,-1,1])
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Positions of ' + str(n) + ' robots at the end of a simulation')
plt.xlim([xlim_min-1,xlim_max+1])
plt.ylim([ylim_min-1,ylim_max+1])
plt.show()

# start graph
plt.figure()

endgabs = []
for i in range(n):
    plt.scatter(pos[i,0,0],pos[i,0,1],label=str(i))
for i in range(len(endpos)):
    for j in range(i+1,len(endpos)):
        if(gabriel(i,j,endpos[i],endpos[j],endpos)):
            plt.plot((endpos[i,0],endpos[j,0]),(endpos[i,1],endpos[j,1]))
            endgabs.append((i,j))
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Graph connections at start of simulation')
plt.xlim([xlim_min-1,xlim_max+1])
plt.ylim([ylim_min-1,ylim_max+1])
plt.legend()
plt.savefig('startpoints_'+str(n)+'.png')
plt.show()

# end graph
plt.figure()
endpos = pos[:,pos.shape[1] - 1,:]
endgabs = []
for i in range(n):
    plt.scatter(pos[i,-1,0],pos[i,-1,1],label=str(i))
for i in range(len(endpos)):
    for j in range(i+1,len(endpos)):
        if(gabriel(i,j,endpos[i],endpos[j],endpos)):
            plt.plot((endpos[i,0],endpos[j,0]),(endpos[i,1],endpos[j,1]))
            endgabs.append((i,j))
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.title('Graph connections at end of simulation')
plt.xlim([xlim_min-1,xlim_max+1])
plt.ylim([ylim_min-1,ylim_max+1])
plt.legend()
plt.savefig('endpoints_'+str(n)+'.png')
plt.show()

# distances with end neighbors
time = np.linspace(0,45,pos.shape[1])
plt.figure()
for i in range(len(endgabs)):
    ri = endgabs[i][0]
    rj = endgabs[i][1]
    dist = pos[ri] - pos[rj]
    dist = np.linalg.norm(dist,axis=1)
    plt.plot(time,dist,label=str(ri) + ' ' + str(rj))

plt.legend()
plt.ylim([-2,10])
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.title('Distance between ending neighbors over course of simulation')
plt.savefig('distancesendpoints_'+str(n)+'.png')
plt.show()

# distances with all neighbors
plt.figure()
l = []
for t in range(pos.shape[1]):
    s = 0
    c = 0
    for i in range(n):
        for j in range(i+1,n):
            ri = pos[i][t]
            rj = pos[j][t]
            if(gabriel(i,j,ri,rj,pos[:,t,:])):
                dist = ri - rj
                dist = np.linalg.norm(dist)
                s += dist
                c += 1
    l.append(s/c)
plt.xlabel('Time (s)')
plt.ylabel('Distance (m)')
plt.title('Average distance between all neighbors at each time step')
plt.ylim([-2,10])
plt.plot(time,l)
plt.savefig('distancesneighbors__'+str(n)+'.png')
plt.show()

# final distance statistics
dataset = []
for experiment in ns:
    alldata = np.load('positionList_expert_'+str(experiment)+'_longer.npy')
    #if(experiment == 12 or experiment == 8):
    #    alldata = alldata[1:]
    #if(experiment == 8):
    #    alldata = alldata[:1]
    allpos = np.zeros((alldata.shape[0],experiment,alldata.shape[1]//experiment,2))
    for i in range(alldata.shape[0]):
        curr = alldata[i]
        for j in range(experiment):
            check = curr[j::experiment]
            allpos[i][j] = check
    alldata = allpos
    dataset.append(alldata)

allm = []
alls = []
index = 0
plt.figure()
for experiment in ns:
    end_positions = dataset[index][:,:,-1,:]
    m = []
    std = []
    for trial in range(len(end_positions)):
        s = []

        sim = end_positions[trial]
        for i in range(experiment):
            for j in range(i+1,experiment):
                ri = sim[i]
                rj = sim[j]
                if(gabriel(i,j,ri,rj,sim)):
                    dist = ri - rj
                    dist = np.linalg.norm(dist)
                    s.append(dist)
        m.append(np.average(s))
        std.append(np.std(s))
    print(m)
    m = np.average(m)
    std = np.sqrt(np.dot(std,std)/len(std))
    allm.append(m)
    alls.append(std)
    index += 1
print(allm,alls)
plt.xlabel('Number of robots')
plt.ylabel('Distance (m)')
plt.title('Average ending distance of neighbors in each experiment')
plt.bar(ns,allm,.4,yerr=alls)
plt.savefig('average_final_distances.png')
plt.show()
