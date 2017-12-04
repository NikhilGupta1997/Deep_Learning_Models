'''
==========
RNN Models
==========
'''

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
import time
import sys
from gru import CustomGRUCell

# Global Variables
EMBEDDING_DIM = 128
HIDDEN_DIM = 64

MODE = str(sys.argv[1]) 	# Cell Type : RNN, LSTM or GRU
EPOCHS = int(sys.argv[2])
LEARNING_RATE = float(sys.argv[3])

''' Map each sequence to corresponding numbered index'''
def prepare_sequence(seq, to_ix):
	idxs = [to_ix[w] if w in to_ix else 0 for w in seq]
	tensor = torch.LongTensor(idxs)
	return autograd.Variable(tensor)

''' Read Data from files '''
def get_data(seq_file, tag_file):
	seqs = [line.split() for line in open(seq_file)]
	tags = [line.split() for line in open(tag_file)]
	return zip(seqs, tags)

''' Train Model on training data '''
def train():
	loss_function = nn.NLLLoss()
	optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
	best_saved_score = 0
	for epoch in range(EPOCHS):
		print "*** Epoch {} ***".format(epoch)
		for i, (sentence, tags) in enumerate(training_data):
			if i % 5000 == 0:
				print "Iteration = {}".format(i)
			model.zero_grad()
			model.hidden = model.init_hidden()
			sentence_in = prepare_sequence(sentence, word_to_ix)
			targets = prepare_sequence(tags, tag_to_ix)
			tag_scores = model(sentence_in)
			loss = loss_function(tag_scores, targets)
			loss.backward()
			optimizer.step()
		test() # Check Validation Accuracy at each train epoch
		torch.save(model.state_dict(), MODE + "_" + str(epoch) + ".txt")
	return model

''' Prediction Accuracy of model on validation dataset '''
def test():
	total, correct = 0, 0
	for sentence, tags in val_data:
		sentence_in = prepare_sequence(sentence, word_to_ix)
		targets = prepare_sequence(tags, tag_to_ix)
		tag_scores = model(sentence_in)
		predicted = torch.max(tag_scores.data,1)[1]
		total += len(tags)
		correct += (predicted == targets.data).sum()
	print "Prediction score = {}/{} = {}%".format(correct, total, float(100 * correct)/float(total))
	return float(correct) / total

''' The Deep Neural Net Model '''
class SequenceTagger(nn.Module):
	def __init__(self, mode, embedding_dim, hidden_dim, vocab_size, tagset_size):
		super(SequenceTagger, self).__init__()
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.model = mode
		self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
		self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
		self.cell = self.init_model()
		self.hidden = self.init_hidden()

	def init_model(self):
		if self.model == 'LSTM':
			mycell = nn.LSTMCell(self.embedding_dim, self.hidden_dim)
		elif self.model == 'GRU':
			mycell = nn.GRUCell(self.embedding_dim, self.hidden_dim)
		elif self.model == 'RNN':
			mycell = nn.RNNCell(self.embedding_dim, self.hidden_dim)
		else:
			raise Exception('Specify one of following cell types: LSTM, GRU, RNN')
		return mycell

	def init_hidden(self):
		if self.model == 'LSTM':
			hidden = (autograd.Variable(torch.zeros(1, self.hidden_dim)),
						autograd.Variable(torch.zeros(1, self.hidden_dim)))
		elif self.model == 'GRU':
			hidden = (autograd.Variable(torch.zeros(1, self.hidden_dim)))
		elif self.model == 'RNN':
			hidden = (autograd.Variable(torch.zeros(1, self.hidden_dim)))
		return hidden

	def forward(self, sentence):
		embeds = self.word_embeddings(sentence)
		output = []
		for i in range(len(sentence)):
			self.hidden = self.cell(embeds[i].view(1, -1), self.hidden)
			output.append(self.hidden[0])
		output = torch.stack(output)
		tag_space = self.hidden2tag(output.view(len(sentence), -1))
		tag_scores = F.log_softmax(tag_space)
		return tag_scores

# Get data from files
training_data= get_data("train_sentences.txt", "train_tags.txt")
val_data = get_data("val_sentences.txt", "val_tags.txt")

# Get index maps initiated on training data
word_to_ix = {}
tag_to_ix = {}
for sent, tags in training_data:
	for word in sent:
		if word not in word_to_ix:
			word_to_ix[word] = len(word_to_ix)
	for tag in tags:
		if tag not in tag_to_ix:
			tag_to_ix[tag] = len(tag_to_ix)

# Define model and Learning parameters
model = SequenceTagger(MODE, EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))

# Train the model
train_start = time.time()
model = train()
print("Total Training Time = {}".format(time.time() - train_start))

# Test the model on the validation set
validation_start = time.time()
test()
print("Total Validation Time = {}".format(time.time() - validation_start))