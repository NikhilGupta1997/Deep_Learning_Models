'''
=======
AlexNet
=======
'''

import torch
import torchvision
from torchvision import transforms, datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import time
import torch.nn.functional as F
import sys

# Global Variables
CLASSES = 35
ROOT = "dataset/"
MODEL = "Original"
DROPOUT = False
RELU = True
OPT = "Paper"

''' AlexNet Model '''
class AlexNet(nn.Module):
	def __init__(self):
		super(AlexNet, self).__init__()
		self.c1 = nn.Conv2d(3, 96, 11, stride = 4, padding = 3)
		self.c2 = nn.Conv2d(96, 256, 5, padding = 2, groups = 2)
		self.c3 = nn.Conv2d(256, 384, 3, padding = 1)
		self.c4 = nn.Conv2d(384, 384, 3, padding = 1, groups = 2)
		self.c5 = nn.Conv2d(384, 256, 3, padding = 1, groups = 2)
		self.f1 = nn.Linear(9216, 256)
		self.f2 = nn.Linear(256, 128)
		self.f3 = nn.Linear(128, CLASSES)

	def activation(self, state):
		if(RELU):
			return F.relu(state)
		else:
			return F.tanh(state)

	def forward(self, state):
		# Apply Convolution Layers
		state = F.max_pool2d(self.activation(self.c1(state)), 3, stride = 2)
		state = F.max_pool2d(self.activation(self.c2(state)), 3, stride = 2)
		state = self.activation(self.c3(state))
		state = self.activation(self.c4(state))
		state = F.max_pool2d(self.activation(self.c5(state)), 3, stride = 2)

		state = state.view(-1, 9216)

		# Apply Fully Connected Layers
		state = self.activation(self.f1(state))
		if(DROPOUT):
			state = F.dropout(state)
		state = self.activation(self.f2(state))
		if(DROPOUT):
			state = F.dropout(state)
		state = self.f3(state)
		return state

''' Read Input from Files '''
def take_input(folder):
	image_transform = transforms.Compose([
		transforms.Scale(256),
		transforms.CenterCrop(256),
		transforms.RandomCrop(224),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor()])
	images = datasets.ImageFolder(root=folder, transform=image_transform)
	data = torch.utils.data.DataLoader(images, batch_size=128, shuffle=True)
	print("File " + folder + " Loaded")
	return data

''' Train on training data '''
def train(model):
	model.train()

	if(OPT == "SGD"): optimizer = optim.SGD(model.parameters(), lr=0.01)
	elif(OPT == "SGD_M"): optimizer = optim.SGD(model.parameters(), lr=0.01,momentum=0.9)
	elif(OPT == "ADAM"): optimizer = optim.Adam(model.parameters(), lr=0.0001)
	else: optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay = 0.0005)
	
	scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor= 0.1)
	criterion = nn.CrossEntropyLoss()	

	prev_val_acc = 0.0 
	for epoch in range(80):
		epoch_start_time = time.time()
		print("*** Epoch {0} ***".format(epoch))
		train_total = 0
		train_correct = 0
		for i, data in enumerate(train_data):
			inputs, labels = data
			labels_old = labels
			if torch.cuda.is_available():
				inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
			else:
				inputs, labels = Variable(inputs), Variable(labels)
			optimizer.zero_grad()
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			predicted = torch.max(outputs, 1)[1]

			train_total += labels.size(0)
			train_correct += (predicted == labels).sum().data.numpy().item(0)
			print(train_correct)

		print("Validating")
		val_acc = test(val_data)
		train_acc = 100.0 * float(correct)/total

		scheduler.step(val_acc)
		
		prev_val_acc = val_acc

		print("* Epoch {} Summary *".format(epoch))
		print("Time Taken = {}").format(time.time() - epoch_start_time)
		print("Val accuracy = {}%".format(val_acc))
		print("Training accuracy = {}%".format(train_acc)) 
		if epoch%10 == 9:
			torch.save(model.state_dict(), ROOT + MODEL + "_" + str(epoch+1) + ".txt")

''' Get prediction values on a dataset '''
def test(dataset):
	model.eval()
	correct = 0
	total = 0
	for i, data in enumerate(dataset):
		inputs, labels = data
		if torch.cuda.is_available():
			inputs = Variable(inputs.cuda())
		else:
			inputs = Variable(inputs)
		outputs = model(inputs)
		predicted = torch.max(outputs, 1)[1]
		total += labels.size(0)
		correct += (predicted.cpu() == labels).sum()
	return float(correct)*100 / total

# Get data from files
train_data = take_input(ROOT + 'train/')
test_data = take_input(ROOT + 'test/')
val_data = take_input(ROOT + 'validation/')

# Define model and Learning parameters
model = AlexNet()
if torch.cuda.is_available():
	print("CUDA is available")
	model = model.cuda()

# Train the model
print("Training AlexNet")
train_start = time.time()
train(model)
print("Total Training Time = {}".format(time.time() - train_start))

# Test the model on the test set
print("Predicting Test Values")
test_acc = test(test_data)
print("Test accuracy = {}%".format(test_acc))

