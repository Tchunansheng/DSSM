import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import *


class CrossEntropyLabelSmooth(nn.Module):
	"""Cross entropy loss with label smoothing regularizer.

	Reference:
	Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
	Equation: y = (1 - epsilon) * y + epsilon / K.

	Args:
		num_classes (int): number of classes.
		epsilon (float): weight.
	"""

	def __init__(self, num_classes, epsilon=0.1):
		super(CrossEntropyLabelSmooth, self).__init__()
		self.num_classes = num_classes
		self.epsilon = epsilon
		self.epsilon = 0.0 #add by ansheng
		self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

	def forward(self, inputs, targets,weight):
		"""
		Args:
			inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
			targets: ground truth labels with shape (num_classes)
		"""
		log_probs = self.logsoftmax(inputs)
		targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
		targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
		# loss = (- targets * log_probs).mean(0).sum()
		loss = (- targets * log_probs).sum(dim=1)
		weight = weight.type_as(loss)
		loss = loss * weight
		loss = loss.mean()
		return loss

class SoftEntropy(nn.Module):
	def __init__(self):
		super(SoftEntropy, self).__init__()
		self.logsoftmax = nn.LogSoftmax(dim=1).cuda()

	def forward(self, inputs, targets,weight):
		log_probs = self.logsoftmax(inputs)
		# loss = (- F.softmax(targets, dim=1).detach() * log_probs).mean(0).sum()
		loss = - F.softmax(targets, dim=1).detach() * log_probs
		loss = loss.sum(dim=1)
		weight = weight.type_as(loss)
		loss = loss * weight
		loss = loss.mean()

		return loss
