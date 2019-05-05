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


class ConvNet(nn.Module):
	def __init__(self):
		super(ConvNet, self).__init__()
		self.layer1 = nn.Sequential(
			# nn.Conv2d(1, 30, kernel_size=3, stride=1, padding=1),
			nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2))
		self.layer2 = nn.Sequential(
			# nn.Conv2d(30, 64, kernel_size=3, stride=1, padding=1),
			nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2))
		self.drop_out = nn.Dropout()
		self.fc1 = nn.Linear(896, 50)
		self.fc2 = nn.Linear(50, 30)

	def forward(self, x):
		# print("--------0:", x.size())
		x = x.resize_((50, 1, 30, 6))
		# x = x.resize_((30, 1, 30, 6))
		out = self.layer1(x)
		# print("-------1:", out.size())
		# out = self.layer2(out)
		# print("-------2:", out.size())
		out = out.reshape(out.size(0), -1)
		# print(out.size())
		# print("-------3:", out.size())
		#out = self.drop_out(out)
		# print("-------4:", out.size())
		out = self.fc1(out)
		# print("-------5:", out.size())
		# out = F.relu(self.fc1(out))

		out = self.fc2(out)
		# print("-------6:", out.size())
		return out


class ConvNetOne(nn.Module):
	def __init__(self):
		super(ConvNetOne, self).__init__()
		self.layer1 = nn.Sequential(
			nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(kernel_size=3, stride=2))
		self.drop_out = nn.Dropout()
		self.fc1 = nn.Linear(6*64, 50)
		self.fc2 = nn.Linear(50, 30)

	def forward(self, x):
		# print("--------0:", x.size())
		x = x.resize_((32, 1, 30, 6))
		print('#'*10, x.size())
		# print("========0:", x.size())
		out = self.layer1(x)
		# print("-------1:", out.size())

		out = out.reshape(out.size(0), -1)
		# print("-------3:", out.size())
		out = self.drop_out(out)
		# print("-------4:", out.size())
		out = self.fc1(out)
		# print("-------5:", out.size())
		# out = F.relu(self.fc1(out))
		out = self.fc2(out)
		# print("-------6:", out.size())
		return out


my_dataset = StateDataset()
train_loader = DataLoader(my_dataset, batch_size=50, shuffle=True, num_workers=1)
# net = Net()
net = ConvNet()

"""
# Train the model
total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(epochs):
	for batch_idx, (data, target) in enumerate(train_loader):
		# Run the forward pass
		net_out = net(data)
		# outputs = model(images)
		loss = criterion3(net_out, torch.max(target, 1)[1])
		# loss = criterion(outputs, labels)
		loss_list.append(loss.item())

		# Backprop and perform Adam optimisation
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# Track the accuracy
		# total = labels.size(0)
		# _, predicted = torch.max(net_out.data, 1)
		# correct = (predicted == labels).sum().item()
		# acc_list.append(correct / total)

		if (batch_idx + 1) % 100 == 0:
			print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
				  .format(epoch + 1, epochs, batch_idx + 1, total_step, loss.item(), (0 / 1) * 100))
			# print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
			#       .format(epoch + 1, num_epochs, batch_idx + 1, total_step, loss.item(), (correct / total) * 100))
"""

if __name__ == "__main__":
	# run the main training loop
	epochs = 5
	log_interval = 400
	net.load_state_dict(torch.load('cnn.ckpt'))
	net.eval()
	with torch.no_grad():
		correct = 0
		total = 0
		print('begin test')
		for batch_idx, (data, target) in enumerate(train_loader):
			print(data)
			data = data.view(-1, data.size()[1] * data.size()[2])
			# data = data.view(-1, 30*6)
			net_out = net(data)
			predicted = torch.max(net_out, 1)[1]
			labels = torch.max(target, 1)[1]
			print("predicted=", predicted)
			print("labels=", labels)
			count = 0
			for i in range(len(predicted)):
				if predicted[i] == labels[i]:
					count += 1
			print('Accuracy: {:.4f} %'.format(100.0 * count / len(predicted)))
			break

