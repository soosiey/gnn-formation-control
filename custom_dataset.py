import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class RobotDataset(Dataset):

    def __init__(self, data1, data2, data3, data4, data5, nA, inW = 100, inH = 100, transform = None):
        self.obs = data1
        self.inW = inW
        self.inH = inH
        c1 = np.zeros((self.obs.shape[0]//nA,nA,self.inW,self.inH))
        jump = c1.shape[0]
        print('Arranging Data')
        for i in tqdm(range(len(c1))):
            #print(obs[i].shape)
            #t1 = self.obs[i].reshape((1000,1000))[::10,::10]
            #t2 = self.obs[jump+i].reshape((1000,1000))[::10,::10]
            #t3 = self.obs[2*jump + i].reshape((1000,1000))[::10,::10]
            for j in range(nA):
                c1[i,j] = self.obs[(j * jump) + i].reshape((self.inW, self.inH))
                #c1[i,1] = self.obs[jump+i].reshape((self.inW, self.inH))
                #c1[i,2] = self.obs[2*jump + i].reshape((self.inW, self.inH))
            #c1[i,0] = t1
            #c1[i,1] = t2
            #c1[i,2] = t3
        self.obs = c1.copy()

        self.gt = data2
        c2 = np.zeros((self.obs.shape[0],nA,2))
        for i in tqdm(range(len(c2))):
            for j in range(nA):
                c2[i,j] = self.gt[(j * jump) + i]
            #c2[i,1] = self.gt[jump+i]
            #c2[i,2] = self.gt[2*jump+i]
        self.gt = c2.copy()

        self.graphs = data3
        c3 = np.zeros((self.obs.shape[0],nA, nA,nA))
        for i in tqdm(range(len(c3))):
            for j in range(nA):
                c3[i,j] = self.graphs[(j * jump) + i].reshape((nA,nA))
            #c3[i,1] = self.graphs[jump+i].reshape((3,3))
            #c3[i,2] = self.graphs[2*jump+i].reshape((3,3))
        self.graphs = c3.copy()

        self.refs = data4
        c4 = np.zeros((self.obs.shape[0],nA,1))
        for i in tqdm(range(len(c4))):
            for j in range(nA):
                c4[i,j,0] = self.refs[(j * jump) + i]
            #c4[i,1,0] = self.refs[jump+i]
            #c4[i,2,0] = self.refs[2*jump+i]
        self.refs = c4.copy()

        self.alphas = data5
        c5 = np.zeros((self.obs.shape[0],nA,1))
        for i in tqdm(range(len(c5))):
            for j in range(nA):
                c5[i,j,0] = self.alphas[(j * jump) + i]
            #c5[i,1,0] = self.alphas[jump+i]
            #c5[i,2,0] = self.alphas[2*jump+i]
        self.alphas = c5.copy()
        self.transform = transform

    def __len__(self):
        return len(self.obs)

    def __getitem__(self,idx):

        if(torch.is_tensor(idx)):
            idx = idx.tolist()

        sample = self.obs[idx]
        gt = self.gt[idx]
        graph = self.graphs[idx]
        refs = self.refs[idx]
        alphas = self.alphas[idx]

        if(self.transform):
            #for i in range(3):
                #s = sample[:,i]
                #m = np.mean(s)
                #std = np.std(s)
                #sample[:,i] = (s - m)/(std + .00001)
            sample = torch.from_numpy(sample).double()
            gt = torch.from_numpy(gt).double()
            graph = torch.from_numpy(graph).double()
            refs = torch.from_numpy(refs).double()
            alphas = torch.from_numpy(alphas).double()
        return {'data':sample, 'graphs':graph,'actions':gt, 'refs':refs, 'alphas':alphas}
