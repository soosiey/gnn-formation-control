import numpy as np
import matplotlib.pyplot as plt



network1 = np.load('positionList/'+'positionList_network_5.npy')[4]
network2 = np.load('positionList/'+'positionList_network_7.npy')[3]
network3 = np.load('positionList/'+'positionList_network_4.npy')[4]




#############5 robots#################################



s = network1.shape
x1 = []
x2 = []
x3 = []
y1 = []
y2 = []
y3 = []
x4 = []
y4 = []
x5 = []
y5 = []
for i in range(s[0]):
    if(i % 5 == 0):
        x1.append(network1[i,0])
        y1.append(network1[i,1])
    elif(i % 5 == 1):
        x2.append(network1[i,0])
        y2.append(network1[i,1])
    elif(i % 5 == 2):
        x3.append(network1[i,0])
        y3.append(network1[i,1])
    elif(i % 5 == 3):
        x4.append(network1[i,0])
        y4.append(network1[i,1])
    elif(i % 5 == 4):
        x5.append(network1[i,0])
        y5.append(network1[i,1])

x1 = np.array(x1)
x2 = np.array(x2)
x3 = np.array(x3)
x4 = np.array(x4)
x5 = np.array(x5)
y1 = np.array(y1)
y2 = np.array(y2)
y3 = np.array(y3)
y4 = np.array(y4)
y5 = np.array(y5)
x1 = np.array([x1[0],x1[-1]])
x2 = np.array([x2[0],x2[-1]])
x3 = np.array([x3[0],x3[-1]])
x4 = np.array([x4[0],x4[-1]])
x5 = np.array([x5[0],x5[-1]])
y1 = np.array([y1[0],y1[-1]])
y2 = np.array([y2[0],y2[-1]])
y3 = np.array([y3[0],y3[-1]])
y4 = np.array([y4[0],y4[-1]])
y5 = np.array([y5[0],y5[-1]])
x = np.stack([x1,x2,x3,x4,x5])
y = np.stack([y1,y2,y3,y4,y5])
sadj = np.zeros((5,5))
for i in range(5):
    for j in range(5):
        if(i == j):
            continue
        else:
            dist = np.sqrt((x[i,0] - x[j,0])**2 + (y[i,0] - y[j,0])**2)
            if(dist <= 2):
                sadj[i,j] = 1
eadj = np.zeros((5,5))
for i in range(5):
    for j in range(5):
        if(i == j):
            continue
        else:
            dist = np.sqrt((x[i,1] - x[j,1])**2 + (y[i,1] - y[j,1])**2)
            if(dist <= 2):
                eadj[i,j] = 1

plt.figure()
plt.scatter(x[:,0],y[:,0],s=80,facecolors='none',edgecolors='blue')
for i in range(5):
    for j in range(i,5):
        if(sadj[i,j] == 1):
            plt.plot(x[(i,j),1],y[(i,j),1],color='black')
plt.title('Starting position of 5 robots')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.savefig('5start.pdf')
plt.show()

plt.figure()
plt.scatter(x[:,1],y[:,1],s=80,facecolors='none',edgecolors='blue')
for i in range(5):
    for j in range(i,5):
        if(eadj[i,j] == 1):
            plt.plot(x[(i,j),1],y[(i,j),1],color='black')
plt.title('Ending position of 5 robots')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.savefig('5end.pdf')
plt.show()

################################7 robots##########################################3
s = network2.shape
x1 = []
x2 = []
x3 = []
y1 = []
y2 = []
y3 = []
x4 = []
y4 = []
x5 = []
y5 = []
x6 = []
y6 = []
x7 = []
y7 = []
for i in range(s[0]):
    if(i % 7 == 0):
        x1.append(network2[i,0])
        y1.append(network2[i,1])
    elif(i % 7 == 1):
        x2.append(network2[i,0])
        y2.append(network2[i,1])
    elif(i % 7 == 2):
        x3.append(network2[i,0])
        y3.append(network2[i,1])
    elif(i % 7 == 3):
        x4.append(network2[i,0])
        y4.append(network2[i,1])
    elif(i % 7 == 4):
        x5.append(network2[i,0])
        y5.append(network2[i,1])
    elif(i % 7 == 5):
        x6.append(network2[i,0])
        y6.append(network2[i,1])
    elif(i % 7 == 6):
        x7.append(network2[i,0])
        y7.append(network2[i,1])
x1 = np.array(x1)
x2 = np.array(x2)
x3 = np.array(x3)
x4 = np.array(x4)
x5 = np.array(x5)
x6 = np.array(x6)
x7 = np.array(x7)
y1 = np.array(y1)
y2 = np.array(y2)
y3 = np.array(y3)
y4 = np.array(y4)
y5 = np.array(y5)
y6 = np.array(y6)
y7 = np.array(y7)
x1 = np.array([x1[0],x1[-1]])
x2 = np.array([x2[0],x2[-1]])
x3 = np.array([x3[0],x3[-1]])
x4 = np.array([x4[0],x4[-1]])
x5 = np.array([x5[0],x5[-1]])
x6 = np.array([x6[0],x6[-1]])
x7 = np.array([x7[0],x7[-1]])
y1 = np.array([y1[0],y1[-1]])
y2 = np.array([y2[0],y2[-1]])
y3 = np.array([y3[0],y3[-1]])
y4 = np.array([y4[0],y4[-1]])
y5 = np.array([y5[0],y5[-1]])
y6 = np.array([y6[0],y6[-1]])
y7 = np.array([y7[0],y7[-1]])
x = np.stack([x1,x2,x3,x4,x5,x6,x7])
y = np.stack([y1,y2,y3,y4,y5,y6,y7])
sadj = np.zeros((7,7))
for i in range(7):
    for j in range(7):
        if(i == j):
            continue
        else:
            dist = np.sqrt((x[i,0] - x[j,0])**2 + (y[i,0] - y[j,0])**2)
            if(dist <= 2):
                sadj[i,j] = 1
eadj = np.zeros((7,7))
for i in range(7):
    for j in range(7):
        if(i == j):
            continue
        else:
            dist = np.sqrt((x[i,1] - x[j,1])**2 + (y[i,1] - y[j,1])**2)
            if(dist <= 2):
                eadj[i,j] = 1

plt.figure()
plt.scatter(x[:,0],y[:,0],s=80,facecolors='none',edgecolors='blue')
for i in range(7):
    for j in range(i,7):
        if(sadj[i,j] == 1):
            plt.plot(x[(i,j),1],y[(i,j),1],color='black')
plt.title('Starting position of 7 robots')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.savefig('7start.pdf')
plt.show()

plt.figure()
plt.scatter(x[:,1],y[:,1],s=80,facecolors='none',edgecolors='blue')
for i in range(7):
    for j in range(i,7):
        if(eadj[i,j] == 1):
            plt.plot(x[(i,j),1],y[(i,j),1],color='black')
plt.title('Ending position of 7 robots')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.savefig('7end.pdf')
plt.show()




################################3 robots##########################################3
s = network3.shape
x1 = []
x2 = []
x3 = []
x4 = []
y1 = []
y2 = []
y3 = []
y4 = []
for i in range(s[0]):
    if(i % 4 == 0):
        x1.append(network3[i,0])
        y1.append(network3[i,1])
    elif(i % 4 == 1):
        x2.append(network3[i,0])
        y2.append(network3[i,1])
    elif(i % 4 == 2):
        x3.append(network3[i,0])
        y3.append(network3[i,1])
    elif(i % 4 == 3):
        x4.append(network3[i,0])
        y4.append(network3[i,1])

x1 = np.array(x1)
x2 = np.array(x2)
x3 = np.array(x3)
x4 = np.array(x4)
y1 = np.array(y1)
y2 = np.array(y2)
y3 = np.array(y3)
y4 = np.array(y4)
x1 = np.array([x1[0],x1[-1]])
x2 = np.array([x2[0],x2[-1]])
x3 = np.array([x3[0],x3[-1]])
x4 = np.array([x4[0],x4[-1]])
y1 = np.array([y1[0],y1[-1]])
y2 = np.array([y2[0],y2[-1]])
y3 = np.array([y3[0],y3[-1]])
y4 = np.array([y4[0],y4[-1]])
x = np.stack([x1,x2,x3,x4])
y = np.stack([y1,y2,y3,y4])
sadj = np.zeros((4,4))
for i in range(4):
    for j in range(4):
        if(i == j):
            continue
        else:
            dist = np.sqrt((x[i,0] - x[j,0])**2 + (y[i,0] - y[j,0])**2)
            if(dist <= 2):
                sadj[i,j] = 1
eadj = np.zeros((4,4))
for i in range(4):
    for j in range(4):
        if(i == j):
            continue
        else:
            dist = np.sqrt((x[i,1] - x[j,1])**2 + (y[i,1] - y[j,1])**2)
            if(dist <= 2):
                eadj[i,j] = 1

plt.figure()
plt.scatter(x[:,0],y[:,0],s=80,facecolors='none',edgecolors='blue')
for i in range(4):
    for j in range(i,4):
        if(sadj[i,j] == 1):
            plt.plot(x[(i,j),1],y[(i,j),1],color='black')
plt.title('Starting position of 4 robots')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.savefig('4start.pdf')
plt.show()

plt.figure()
plt.scatter(x[:,1],y[:,1],s=80,facecolors='none',edgecolors='blue')
for i in range(4):
    for j in range(i,4):
        if(eadj[i,j] == 1):
            plt.plot(x[(i,j),1],y[(i,j),1],color='black')
plt.title('Ending position of 4 robots')
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.savefig('4end.pdf')
plt.show()


