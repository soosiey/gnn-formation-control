import numpy as np
import matplotlib.pyplot as plt

def get_average(arr):
    mean = np.mean(arr,axis=1)
    diff = np.subtract(arr,mean.reshape((arr.shape[0],1,arr.shape[2])))
    norm = np.linalg.norm(diff,axis=2)
    avgs = np.mean(norm,axis=1)
    average = np.mean(avgs)
    return average

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
def get_end():
    ynet = []
    yexpert = []
    for i in range(8,9):

        network = np.load('positionList_network_'+str(i)+'.npy')
        expert = np.load('positionList_expert_'+str(i)+'.npy')
        ynet.append(get_average(network))
        yexpert.append(get_average(expert))
def main():

    colors = ['blue','orange','green','red','purple','brown']
    x = [[] for i in range(6)]
    xn = [[] for i in range(6)]
    for i in range(3,9):
        xn[i - 3] = get_running_average(np.load('positionList_network_'+str(i)+'.npy'),i)
        x[i - 3] = get_running_average(np.load('positionList_expert_'+str(i)+'.npy'),i)

    plt.plot()
    for i in range(3,9):
        plt.plot(xn[i - 3],label=str(i),color=colors[i-3],ls='dotted')
        plt.plot(x[i - 3],label=str(i),color=colors[i-3])
    plt.legend()
    plt.show()
main()

