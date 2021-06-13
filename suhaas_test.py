import torch
import numpy as np
from graphs.models import suhaas_model
from torch.utils.data import DataLoader
from torchvision import transforms
import custom_dataset
import torch.nn as nn
from tqdm import tqdm

def main():

    np.random.seed(0)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    x = np.array([[0,0,1],[1,0,0],[0,1,0]])
    c = np.zeros((54000,3,3))
    for i in range(54000):
        c[i] = x
    x = c
    x = torch.from_numpy(x)

    transform = transforms.Compose([transforms.ToTensor()])

    data = np.load('data001_0.npz')
    inputs = data['observations']
    actions = data['actions']
    trin = inputs[:43200]
    trac = actions[:43200]
    trg = x[:43200]
    tein = inputs[43200:]
    teac = actions[43200:]
    teg = x[43200:]

    lr = .0001
    epochs = 100

    trainset = custom_dataset.RobotDataset(trin,trac,trg,inW = 50,inH = 50, transform = transform)
    testset = custom_dataset.RobotDataset(tein,teac,teg,inH = 50, inW = 50, transform = transform)

    trainloader = DataLoader(trainset, batch_size = 16, shuffle = True, drop_last = True)
    testloader = DataLoader(testset, batch_size = 16, shuffle = True, drop_last = True)

    model = suhaas_model.DecentralPlannerNet(inW=50,inH=50).double()
    print(model)
    model = model.to('cuda')
    x = x.to('cuda')

    criterion = nn.MSELoss()
    optimizer = torch.optim.RMSprop([p for p in model.parameters() if p.requires_grad], lr = lr)

    for epoch in range(epochs + 1):
        model.train()

        epoch_loss = 0
        total = 0
        for i,batch in enumerate(tqdm(trainloader)):
            inputs = batch['data'].to('cuda')
            labels = batch['actions'].to('cuda')
            x = batch['graphs'][:,:,:,:].to('cuda')
            model.addGSO(x)
            optimizer.zero_grad()
            outs = model(inputs)
            loss = criterion(outs[0],labels[:,0])
            for i in range(1,3):
                loss += criterion(outs[i],labels[:,i])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            total +=  inputs.size(0)

        print('Average training loss for epoch',epoch,'is', epoch_loss / total)

        if epoch % 10 == 0:
            model.eval()
            epoch_loss = 0
            total = 0

            for i,batch in enumerate(tqdm(testloader)):
                inputs = batch['data'].to('cuda')
                labels = batch['actions'].to('cuda')
                model.addGSO(x)
                optimizer.zero_grad()
                outs = model(inputs)
                loss = criterion(outs[0],labels[:,0])
                for i in range(1,3):
                    loss += criterion(outs[i],labels[:,i])
                epoch_loss += loss.item()
                total += inputs.size(0)
            print('Average testing loss for epoch', epoch, 'is', epoch_loss / total)

if __name__ == '__main__':
    main()
