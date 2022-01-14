import torch
import torch.nn as nn
from math import sqrt

class MySCN(nn.Module):
    def __init__(self):
        super(MySCN, self).__init__()
        self.W1 = nn.Conv2d(40, 128, 3, 1, 1, bias=False)
        self.S1 = nn.Conv2d(128, 40, 3, 1, 1, groups=1, bias=False)
        self.S2 = nn.Conv2d(40, 128, 3, 1, 1, groups=1, bias=False)
        self.shlu = nn.ReLU(True)


    def forward(self, input):
        x1 = input[:,range(0,40),:,:]
        event_input = input[:,range(40,80),:,:]

        x1 = torch.mul(x1,event_input)
        z = self.W1(x1)
        tmp = z
        for i in range(25):
            ttmp = self.shlu(tmp)
            x = self.S1(ttmp)
            x = torch.mul(x,event_input)
            x = torch.mul(x,event_input)
            x = self.S2(x)
            x=ttmp-x
            tmp = torch.add(x, z)
        c = self.shlu(tmp)
        return c

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.scn = nn.Sequential(MySCN())
        self.image_d=nn.Conv2d(in_channels=1, out_channels=40, kernel_size=1, stride=1, padding=0, bias=False)
        self.event_c1=nn.Conv2d(in_channels=40, out_channels=40, kernel_size=1, stride=1, padding=0, bias=False)
        self.event_c2=nn.Conv2d(in_channels=40, out_channels=40, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.W2=nn.Conv2d(in_channels=128, out_channels=40, kernel_size=3, stride=1, padding=1, bias=False)
        self.endconv=nn.Conv2d(in_channels=40, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
    
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x, eventstream, t): 
        x1=self.image_d(x)

        event_1 = eventstream[:,range(0,40),:,:]
        event_out_1=self.event_c1(event_1)
        event_out_1=torch.sigmoid(event_out_1)
        event_out_1=self.event_c2(event_out_1)
        E_1=torch.sigmoid(event_out_1)

        event_2 = eventstream[:,range(40,80),:,:]
        event_out_2=self.event_c1(event_2)
        event_out_2=torch.sigmoid(event_out_2)
        event_out_2=self.event_c2(event_out_2)
        E_2=torch.sigmoid(event_out_2)

        event_out = (1-t) * E_1 + t * E_2

        scn_input=torch.cat([x1,event_out],1)
        
        out = self.scn(scn_input)
        out=self.W2(out)
        out=self.endconv(out)
        return out
