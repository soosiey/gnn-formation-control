import torch
import numpy as np
from graphs.models import suhaas_model
from torch.utils.data import DataLoader
from torchvision import transforms
import custom_dataset
import torch.nn as nn
from tqdm import tqdm


class Agent():

    def __init__(self, criterion = 'mse', optimizer = 'rms', inW = 100, inH = 100,  nA = 3, lr = .01):
        self.points_per_ep = None
        self.nA = nA
        self.inW = inW
        self.inH = inH
        self.model = suhaas_model.DecentralPlannerNet(nA = self.nA, inW = self.inW, inH = self.inH).double()
        self.model = self.model.to('cuda')
        self.lr = lr
        if(criterion == 'mse'):
            self.criterion = nn.MSELoss()
        if(optimizer == 'rms'):
            self.optimizer = torch.optim.RMSprop([p for p in self.model.parameters() if p.requires_grad], lr = self.lr)
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.epoch = -1
        self.lr_schedule = {0:.0001, 10:.0001, 20:.0001}
        self.currentAgent = -1
    def test(self, x, S, refs, alphas):

        self.currentAgent += 1
        self.currentAgent = self.currentAgent % 3
        xin = np.zeros((1, 1, self.inW, self.inH))

        x = x[0]
        #x = x.reshape((1000,1000))[::10,::10]
        x = x.reshape((self.inW, self.inH))
        xin[0,0] = x
        xin = torch.from_numpy(xin).double()
        #xin = xin.unsqueeze(0)
        #xin = xin.unsqueeze(0)
        xin = xin.to('cuda')
        S = np.array(S)
        S = S.reshape((3,3))
        S = np.roll(S,-1*self.currentAgent,axis=0)
        S = torch.from_numpy(S)
        S = S.unsqueeze(0)
        S = S.to('cuda')

        r = np.zeros((1,3,1))
        r[0,0,0] = refs
        r = torch.from_numpy(r).double()
        r = r.to('cuda')

        a = np.zeros((1,3,1))
        a[0,0,0] = alphas
        a = torch.from_numpy(a).double()
        a = a.to('cuda')
        self.model.eval()
        self.model.addGSO(S)
        outs = [self.model.forward_one(xin,r,a)[0]]
        return outs

    def train(self, data):
        """
        datalist[0].d['actions', 'graph', 'observations']
        """
        self.epoch += 1
        if(self.epoch in self.lr_schedule.keys()):
            for g in self.optimizer.param_groups:
                g['lr'] = self.lr_schedule[self.epoch]
        actions = data[0].d['actions']
        inputs = data[0].d['observations']
        graphs = data[0].d['graph']
        refs = data[0].d['obs2'][:,1]
        alphas = data[0].d['obs2'][:,2]
        #np.save('actions.npy', actions)
        #np.save('inputs.npy', inputs)
        #np.save('graphs.npy', graphs)
        trainset = custom_dataset.RobotDataset(inputs,actions,graphs,refs,alphas,inW = self.inW, inH = self.inH,transform = self.transform)
        trainloader = DataLoader(trainset, batch_size = 16, shuffle = True, drop_last = True)
        self.model.train()
        total_loss = 0
        total = 0
        for i,batch in enumerate(tqdm(trainloader)):
            inputs = batch['data'].to('cuda')
            S = batch['graphs'][:,0,:,:].to('cuda')
            actions = batch['actions'].to('cuda')
            refs = batch['refs'].to('cuda')
            alphas = batch['alphas'].to('cuda')
            self.model.addGSO(S)
            self.optimizer.zero_grad()
            outs = self.model(inputs,refs,alphas)
            loss = self.criterion(outs[0], actions[:,0])
            for i in range(1,self.nA):
                loss += self.criterion(outs[i], actions[:,i])
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total += inputs.size(0)*3
        print('Average training loss:', total_loss / total)
        return total_loss / total

    def save(self,pth):
        torch.save(self.model.state_dict(), 'models/' + pth)
