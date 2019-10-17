
import torchvision
from torch import nn
import torch
import torch.nn.functional as F

from Model_Parts import *


#create goal conditioned policies

class Causal_Induction_Model(nn.Module):
	def __init__(self,batch_size,n_channels,n_variables):
		super(Causal_Induction_Model, self).__init__()
		
		self.C = torch.zeros(batch_size,n_variables,n_variables)
		self.observation_encoder = Observation_Encoder_F(n_channels,n_variables)
		self.transition_encoder = Transition_Encoder(n_channels+1,n_variables)
		self.edge_encoder = Edge_Encoder()
		self.c_encoder = C_Encoder(n_variables)
		
		
	def forward(self, x , a): #B*A*C*H*W
		
		batch_size = x.size(0)
		H = x.size(1)
		
		for i in range(H-1):
			x1 = self.observation_encoder(x[:,i,:,:,:])
			x2 = self.observation_encoder(x[:,i+1,:,:,:])
			R = x2-x1
			T = torch.cat((R,a[:,i]),1)
			E = self.transition_encoder(T)
			delata_c = self.edge_encoder(E)
			self.C = self.C + delata_c
			
		E = self.c_encoder(self.C)
		self.C = self.C + self.edge_encoder(E)
		
		return self.C # B*N*N loss = nn.MSELoss() about C
			

"""
Q = torch.rand(12,5,3,32,32,requires_grad=True)
a = torch.rand(12,5,1)
print(Q.shape)
print(a.shape)
model = Causal_Induction_Model(12,3,5)
print(model)
print(model(Q,a))
print(model(Q,a).shape)
"""
