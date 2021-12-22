import numpy as np
import matplotlib.pyplot as plt

d = np.load('data/data070_0.npz')

d = d['observations']

s5 = d[100:105]
s10 = d[2000:2005]
s15 = d[4000:4005]
for i in range(5):
    start = i*810
    astart = start+100
    bstart = start+(270*1)+100
    cstart = start+(270*2)+100
    a = d[astart].reshape((100,100))
    b = d[bstart].reshape((100,100))
    c = d[cstart].reshape((100,100))
    plt.figure()
    plt.imshow(a,cmap='gray')
    plt.savefig('5_'+str(i)+'.png')
    plt.figure()
    plt.imshow(b,cmap='gray')
    plt.savefig('10_'+str(i)+'.png')
    plt.figure()
    plt.imshow(c,cmap='gray')
    plt.savefig('15_'+str(i)+'.png')
