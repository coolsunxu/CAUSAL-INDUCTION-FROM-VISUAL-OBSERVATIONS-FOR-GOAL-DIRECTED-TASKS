
import torchvision
from torch import nn
import torch

import torch.nn.functional as F


#create CONV+RELU+POOL
class conv_br(nn.Module):
	def __init__(self, in_ch, out_ch):
		super(conv_br, self).__init__()
		self.conv = nn.Sequential(nn.Conv2d(in_ch, out_ch, 3, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2)
		)

	def forward(self, x):
		x = self.conv(x)
		return x
		
#create full
class fout_relu(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(fout_relu, self).__init__()
        self.fullout = nn.Sequential(
			nn.Linear(in_ch, out_ch),
			nn.ReLU(inplace=True),
			nn.Dropout(0.3)
		)

    def forward(self, x):
        x = self.fullout(x)
        return x
        
#create full
class fout(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(fout, self).__init__()
        self.fullout = nn.Sequential(
			nn.Linear(in_ch, out_ch)
		)

    def forward(self, x):
        x = self.fullout(x)
        return x
        


#create Observation_Encoder in causal induction model

class Observation_Encoder_F(nn.Module):
	def __init__(self,n_channels,n_variables):
		super(Observation_Encoder_F, self).__init__()
		
		self.conv1 = conv_br(n_channels, 8)
		self.conv2 = conv_br(8, 16)
		self.conv3 = conv_br(16, 32)
		
		self.outc = fout(512, n_variables)
		
	def forward(self, x):
		
		batch_size = x.size(0)
		
		x1 = self.conv1(x)
		x2 = self.conv2(x1)
		x3 = self.conv3(x2)
		x4 = x3.view(batch_size, -1)
		out = self.outc(x4)
		
		return out
		
		
#create Transition Encoder in causal induction model
	
class Transition_Encoder(nn.Module):
	def __init__(self,n_channels,n_variables):
		super(Transition_Encoder, self).__init__()
		
		self.fout1 = fout_relu(6, 1024)
		self.fout2 = fout_relu(1024, 512)
		self.fout3 = fout(512, n_variables)
		
		
	def forward(self, x):
		
		batch_size = x.size(0)
		
		x1 = self.fout1(x)
		x2 = self.fout2(x1)
		x3 = self.fout3(x2)
		out = F.softmax(x3,1)
		
		return out
	
#create Edge Encoder in causal induction model
	
class Edge_Encoder(nn.Module):
	def __init__(self):
		super(Edge_Encoder, self).__init__()
		
	def forward(self, x):
		
		batch_size = x.size(0)
		
		x1 = torch.sigmoid(x)
		x2 = torch.unsqueeze(x1,1) # delta e
		x3 = torch.unsqueeze(x,2) # a
		out = torch.stack([x3[i].mm(x2[i]) for i in range(batch_size)],0)

		return out	
		
#create Edge Encoder in causal induction model
	
class C_Encoder(nn.Module):
	def __init__(self,n_variables):
		super(C_Encoder, self).__init__()
		
		self.fout1 = fout(n_variables*n_variables, n_variables)
		
	def forward(self, x):
		
		batch_size = x.size(0)
		x1 = x.view(batch_size, -1)
		out = self.fout1(x1)
		
		return out

