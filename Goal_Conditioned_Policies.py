import torchvision
from torch import nn
import torch
import torch.nn.functional as F

from Model_Parts import *
from Causal_Induction_Model import *

#create goal conditioned policies

class Goal_Conditioned_Policies(nn.Module):
	def __init__(self,batch_size,n_channels,n_variables,s=None,a=None): # n_channels = 3 + 3 = 6 not 3
		super(Goal_Conditioned_Policies, self).__init__()
		if s is None or a is None:
			self.C = torch.rand(batch_size,n_variables,n_variables) # just for test~
		else :
			self.causal_induction_model = Causal_Induction_Model(batch_size,n_channels/2,n_variables)
			self.causal_induction_model.eval()
			self.C = self.causal_induction_model(s,a) #B*A*C*H*W  (s1,a1,s2,a2,......)
		
		self.observation_encoder = Observation_Encoder_F(n_channels,128) #image encoded
		self.fout1 = fout(128,n_variables)
		self.fout2 = fout(n_variables,128)
		
		#set three fully layer
		self.fout3 = fout(256,64)
		self.fout4 = fout(64,64)
		self.fout5 = fout(64,64)
		
		#out
		self.out = fout(64,n_variables)
		
	def forward(self, x): #B*C*H*W C=3+3
		
		batch_size = x.size(0)
		x1 = self.observation_encoder(x)
		x2 = self.fout1(x1)
		a = torch.unsqueeze(x2,2) # a
		e = torch.stack([self.C[i].mm(a[i]) for i in range(batch_size)],0)
		x3 = self.fout2(e.view(batch_size,-1))
		x4 = torch.cat((x1,x3),1)
		
		x5 = self.fout3(x4)
		x6 = self.fout4(x5)
		x7 = self.fout5(x6)
		
		out = F.softmax(self.out(x7),1)
		return out # B*N  loss = nn.CrossEntropyLoss()  about action


"""
Q = torch.rand(12,6,32,32,requires_grad=True)

print(Q.shape)
model = Goal_Conditioned_Policies(12,6,5)
print(model)
print(model(Q).shape)
"""
