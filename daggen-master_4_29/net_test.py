import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class StateDataset(Dataset):
	"""state dataset"""
	def __init__(self, transform=None):
		# a = np.loadtxt('data_state.txt', dtype=np.float32)
		# b = np.loadtxt('data_label.txt', dtype=np.int64)
		a = np.loadtxt('heft_data_state.txt', dtype=np.float32)
		b = np.loadtxt('heft_data_label.txt', dtype=np.int64)
		nstate = torch.from_numpy(a)
		states = nstate.view(-1, 30, 6)
		labels = torch.from_numpy(b)
		self.states = states
		self.labels = labels
		self.transform = transform

	def __len__(self):
		return len(self.states)

	def __getitem__(self, idx):
		state = self.states[idx]
		label = self.labels[idx]
		sample = {'state': state, 'label': label}
		if self.transform:
			sample = self.transform(sample)
		return state, label


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(30 * 6, 100)
		self.fc2 = nn.Linear(100, 50)
		self.fc3 = nn.Linear(50, 30)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		# return F.log_softmax(x)
		return x


my_dataset = StateDataset()
train_loader = DataLoader(my_dataset, batch_size=1, shuffle=True, num_workers=1)

"""
learning_rate = 0.001
net = Net()
# create a stochastic gradient descent optimizer
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
# create a loss function negative log likelihood loss
criterion = nn.NLLLoss()
criterion2 = nn.MSELoss(reduction='sum')
criterion3 = nn.CrossEntropyLoss()
# run the main training loop
epochs = 20
log_interval = 400

for epoch in range(epochs):
	for batch_idx, (data, target) in enumerate(train_loader):
		# print (data.size(),target.size())
		# data, target = Variable(data), Variable(target)
		# print (data.size(),target.size())
		#  resize data from (batch_size, 1, 28, 28) to (batch_size, 28*28)
		data = data.view(-1, 30*6)
		# print (data.size(),target.size())
		optimizer.zero_grad()
		net_out = net(data)
		# print(torch.max(target,1)[1])
		loss = criterion3(net_out, torch.max(target,1)[1])
		# print('net_out:{}'.format(net_out))
		# print('target:{}'.format(torch.max(target,1)[1]))
		# loss = criterion(net_out, target)
		# loss = criterion3(net_out,target)
		loss.backward()
		optimizer.step()
		if batch_idx % log_interval == 0:
			# print('batch_idx:{} len_data:{}'.format(batch_idx,len(data)))
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
					epoch, batch_idx * len(data), len(train_loader.dataset),
						   100. * batch_idx / len(train_loader), loss.item()))
		# print(torch.max(net_out, 1)[1])
		# print(torch.max(target, 1)[1])
torch.save(net.state_dict(), 'mynet.ckpt')
"""

net = Net()
net.load_state_dict(torch.load('mynet.ckpt'))
net.eval()
with torch.no_grad():
	correct = 0
	total = 0
	print('begin test')
	for batch_idx,  (data, target) in enumerate(train_loader):
		print(data)
		data = data.view(-1, 30*6)
		net_out = net(data)
		predicted = torch.max(net_out, 1)[1]
		labels = torch.max(target, 1)[1]
		print("predicted=", predicted)
		print("labels=", labels)
		break
