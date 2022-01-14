import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn import init

########################################################################
#########################################################################


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.DIT1 = nn.Conv2d(32, 128, 3, 1, 1, bias=False)
        self.DI1 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
        self.DIT2 = nn.Conv2d(32, 128, 3, 1, 1, bias=False)
        self.shlu = nn.ReLU(True)
        self.trans=nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.trans2=nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.event_c1=nn.Conv2d(in_channels=40, out_channels=40, kernel_size=1, stride=1, padding=0, bias=False)
        self.event_c2=nn.Conv2d(in_channels=40, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.shu1=nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.shu2=nn.Conv2d(in_channels=128, out_channels=512, kernel_size=3, stride=1, padding=1, bias=False)
        self.endconv=nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.ps1=nn.PixelShuffle(2)
        self.ps2=nn.PixelShuffle(2)

        '''for p in self.parameters():
                    p.requires_grad=False'''

        self.DET1 = nn.Conv2d(32, 128, 3, 1, 1, bias=False)
        self.DE1 = nn.Conv2d(128, 32, 3, 1, 1, bias=False)
        self.DET2 = nn.Conv2d(32, 128, 3, 1, 1, bias=False)
        self.lam = Parameter(torch.Tensor(1))
        init.constant_(self.lam,0)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, x, event_frame, t): 
        Y=self.trans(x)

        event_1 = event_frame[:,range(0,40),:,:]
        event_out_1=self.event_c1(event_1)
        event_out_1=torch.sigmoid(event_out_1)
        event_out_1=self.event_c2(event_out_1)
        E_1=torch.sigmoid(event_out_1)

        event_2 = event_frame[:,range(40,80),:,:]
        event_out_2=self.event_c1(event_2)
        event_out_2=torch.sigmoid(event_out_2)
        event_out_2=self.event_c2(event_out_2)
        E_2=torch.sigmoid(event_out_2)

        E = (1-t) * E_1 + t * E_2



        if self.training:
            e_nonoise = event_frame[:,range(80,120),:,:]
            event_out_n=self.event_c1(e_nonoise)
            event_out_n=torch.sigmoid(event_out_n)
            event_out_n=self.event_c2(event_out_n)
            E_nonoise=torch.sigmoid(event_out_n)

            e_nonoise_f = event_frame[:,range(120,160),:,:]
            event_out_n_f=self.event_c1(e_nonoise_f)
            event_out_n_f=torch.sigmoid(event_out_n_f)
            event_out_n_f=self.event_c2(event_out_n_f)
            E_nonoise_f=torch.sigmoid(event_out_n_f)

            E_nonoise = (1-t) * E_nonoise + t * E_nonoise_f

        y1 = torch.mul(Y, E)
        y1 = self.DIT1(y1)
        alpha = self.shlu(y1)
        
        i_add = torch.mul(Y,self.DI1(alpha))
        e1 = self.lam * E + i_add
        e1 = self.DET1(e1)
        beta = self.shlu(e1)
        for i in range(15):
            en = self.DE1(beta)
            y1 = torch.mul(Y, en)
            #y1 = torch.mul(Y, E)
            y1 = self.DIT1(y1)
            y = self.DI1(alpha)
            en_s = torch.mul(en, en)
            y = torch.mul(y, en_s)
            #y = torch.mul(y, E)
            #y = torch.mul(y, E)
            y = self.DIT2(y)
            yr = alpha + 0.01*y1 - 0.01*y
            
            In = self.DI1(alpha)
            i_add = torch.mul(Y, In)
            e1 = self.lam * E + i_add
            e1 = self.DET1(e1)
            e = self.DE1(beta)
            elam = self.lam + torch.mul(In, In)
            e = torch.mul(e, elam)
            e = self.DET2(e)
            er = beta + 0.01*e1 - 0.01*e
            
            alpha = self.shlu(yr)
            beta = self.shlu(er)

        out=self.shu1(alpha)
        out=self.ps1(out)
        out=self.shu2(out)
        out=self.ps2(out)
        out=self.endconv(out)
        if self.training:
            return out, self.trans2(self.DI1(alpha)), self.DE1(beta), E_nonoise
        else:
            return out