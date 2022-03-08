import numpy as np
import matplotlib.pyplot as plt

def get_average(arr,n):
    mean_total = 0
    st_total = 0
    for i in range(arr.shape[0]):
        current = arr[i,-n:,:]
        m = np.mean(current,axis=0)
        d = np.subtract(current,m.reshape((1,2)))
        d = d * d
        s = np.sum(d,axis=1)
        s = np.sqrt(s)
        m = np.mean(s)
        st = np.std(s)
        mean_total += m
        st_total += (st*st)
    mean_total = mean_total / arr.shape[0]
    st_total = st_total / arr.shape[0]
    st_total = np.sqrt(st_total)
    return mean_total,st_total

def get_running_average(arr,n):

    y = []

    timesteps = arr.shape[1]//n
    for i in range(timesteps):
        mean_total = 0
        for j in range(arr.shape[0]):
            start = i * n
            current = arr[j,start:start+n]
            m = np.mean(current,axis=0)
            d = np.subtract(current,m.reshape((1,2)))
            d = d * d
            s = np.sum(d,axis=1)
            s = np.sqrt(s)
            m = np.mean(s)
            mean_total += m
        mean_total = mean_total / arr.shape[0]
        y.append(mean_total)
    return y

def checkmove(arr):
    moving = False
    for i in range(1,len(arr)):
        if(abs(arr[i] - arr[i-1]) > .001):
            moving = True
    return moving

def get_convergence(arr,n):
    window_size = n*10
    ct_avg = 0
    ct_count = 0
    for i in range(10):
        current = arr[i]
        convergence = False
        tf = None
        for j in range(n*100,current.shape[0] - window_size,n):
            if(convergence):
                break
            window = current[j:j+window_size]
            count = 0
            for k in range(n):
                robot_window = window[k::n]
                check = []
                for t in range(robot_window.shape[0]):
                    d = (robot_window[t,0]**2 + robot_window[t,1]**2)**(.5)
                    check.append(d)
                m = checkmove(check)
                if(not m):
                    count += 1
            if(count >= n):
                convergence = True
                tf = j//n
        if(type(tf) == type(None)):
            continue
        else:
            ct_avg += tf
            ct_count += 1
    if(ct_count != 0):
        return ct_avg // ct_count
    else:
        return -1
def main():

    colors = ['blue','orange','green','red','purple','brown']
    x = [[] for i in range(6)]
    xn = [[] for i in range(6)]


    for i in range(3,9):
        expert = np.load('positionList/'+'positionList_network_'+str(i)+'.npy')
        network = np.load('positionList/'+'positionList_expert_'+str(i)+'.npy')
        xn[i - 3] = get_running_average(np.load('positionList_network_'+str(i)+'.npy'),i)
        x[i - 3] = get_running_average(np.load('positionList_expert_'+str(i)+'.npy'),i)

    plt.figure()
    xaxis = np.linspace(0,15,300)
    for i in range(3,9):
        plt.plot(xaxis,xn[i - 3],label=str(i),color=colors[i-3],ls='dotted')
        plt.plot(xaxis,x[i - 3],label=str(i),color=colors[i-3])
    plt.xlabel('Time (s)')
    plt.ylabel('Average distance from center of mass of formation')
    plt.title('Average distance from center of formation during episode')
    plt.legend()
    plt.savefig('time_graph.png')
    plt.show()

    network_means = []
    expert_means = []
    network_std = []
    expert_std = []
    ticks = np.arange(3,9)
    plt.figure()
    for i in range(3,9):
        expert = np.load('positionList/'+'positionList_network_'+str(i)+'.npy')
        network = np.load('positionList/'+'positionList_expert_'+str(i)+'.npy')
        em,es = get_average(expert,i)
        nm,ns = get_average(network,i)
        network_means.append(xn[i-3][-1])
        expert_means.append(x[i-3][-1])
        network_std.append(ns)
        expert_std.append(es)
    plt.bar(ticks,expert_means,.4,yerr=expert_std,label='expert')
    plt.bar(ticks+.5,network_means,.4,yerr=network_std,label='network')
    plt.xticks(ticks+.5/2,ticks)
    plt.xlabel('Number of robots')
    plt.ylabel('Distance from center of formation')
    plt.title('Ending distance from center of formation')
    plt.legend()
    plt.savefig('distance_graph.png')
    plt.show()

    cns = []
    ces = []
    plt.figure()
    for i in range(3,9):
        #plt.figure()
        expert = np.load('positionList/'+'positionList_network_'+str(i)+'.npy')
        network = np.load('positionList/'+'positionList_expert_'+str(i)+'.npy')
        cn = get_convergence(network,i)
        ce = get_convergence(expert,i)
        #plt.plot(xn[i - 3],color=colors[i-3],ls='dotted',label='network')
        #plt.plot(x[i - 3],color=colors[i-3],label='expert')
        if(cn == -1):
            cns.append(0)
        else:
            cns.append(cn / 20)
        if(ce == -1):
            ces.append(0)
        else:
            ces.append(ce / 20)
            #plt.vlines(cn,0,3,ls='dotted',color=colors[i-3])
    plt.bar(ticks,ces,.4,label='expert')
    plt.bar(ticks+.5,cns,.4,label='network')
            #plt.vlines(ce,0,3,color=colors[i-3])

    plt.xticks(ticks+.5/2,ticks)
    plt.xlabel('Number of robots')
    plt.ylabel('Convergence time (s)')
    plt.title('Convergence time')
    plt.legend()
    plt.savefig('convergence_time_all.png')
    plt.show()
main()

