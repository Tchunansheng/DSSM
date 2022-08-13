from __future__ import absolute_import

import torch
from torch import nn
import torch.nn.functional as F


def cal_triplet_from_an_ap(ap,an,margin,weight):
	loss = 0.0
	for p, n, w in zip(ap,an,weight):
		loss += max(0.0,p+margin-n) * w
	return loss/ap.size(0)

def euclidean_dist(x, y):
	m, n = x.size(0), y.size(0)
	xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
	yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
	dist = xx + yy
	dist.addmm_(1, -2, x, y.t())
	dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
	return dist

def cosine_dist(x, y):
	bs1, bs2 = x.size(0), y.size(0)
	frac_up = torch.matmul(x, y.transpose(0, 1))
	frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
	            (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
	cosine = frac_up / frac_down
	return 1-cosine

def _batch_hard(mat_distance, mat_similarity, indice=True):
	sorted_mat_distance, positive_indices = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1, descending=True)
	hard_p = sorted_mat_distance[:, 0]
	hard_p_indice = positive_indices[:, 0]
	sorted_mat_distance, negative_indices = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
	hard_n = sorted_mat_distance[:, 0]
	hard_n_indice = negative_indices[:, 0]
	if(indice):
		return hard_p, hard_n, hard_p_indice, hard_n_indice
	return hard_p, hard_n

class TripletLoss(nn.Module):
	'''
	Compute Triplet loss augmented with Batch Hard
	Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
	'''

	def __init__(self, margin, normalize_feature=True):
		super(TripletLoss, self).__init__()
		self.margin = margin
		self.normalize_feature = normalize_feature
		self.margin_loss = nn.MarginRankingLoss(margin=margin).cuda()

	def forward(self, emb, label,weight):
		if self.normalize_feature:
			# equal to cosine similarity
			emb = F.normalize(emb)
		mat_dist = euclidean_dist(emb, emb)
		# mat_dist = cosine_dist(emb, emb)
		assert mat_dist.size(0) == mat_dist.size(1)
		N = mat_dist.size(0)
		mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()

		dist_ap, dist_an, hard_p_indice, hard_n_indice = _batch_hard(mat_dist, mat_sim)

		ensemble_weight = []
		for index, (ap_index,an_index) in enumerate(zip(hard_p_indice,hard_n_indice)):
			temp_weight = (2*weight[index] + weight[ap_index] + weight[an_index])/4.0
			ensemble_weight.append(temp_weight)
		ensemble_weight = torch.tensor(ensemble_weight)
		ensemble_weight = ensemble_weight.float().cuda()
		loss = cal_triplet_from_an_ap(dist_ap,dist_an,self.margin,ensemble_weight)
		return loss

		#
		# ensemble_weight_ap=[]
		# ensemble_weight_an=[]
		# for index, (ap_index,an_index) in enumerate(zip(hard_p_indice,hard_n_indice)):
		# 	temp_weight_ap = weight[index] + weight[ap_index] #* weight[an_index]
		# 	temp_weight_an = weight[index] + weight[an_index]
		# 	ensemble_weight_ap.append(temp_weight_ap)
		# 	ensemble_weight_an.append(temp_weight_an)
		#
		# ensemble_weight_an = torch.tensor(ensemble_weight_an)
		# ensemble_weight_an = ensemble_weight_an.float().cuda()
		#
		# ensemble_weight_ap = torch.tensor(ensemble_weight_ap)
		# ensemble_weight_ap = ensemble_weight_ap.float().cuda()
		#
		# dist_an = torch.exp(dist_an)
		# dist_ap = torch.exp(dist_ap)
		# dist_an = dist_an * ensemble_weight_an
		# dist_ap = dist_ap * ensemble_weight_ap
		#
		# loss = torch.sum(-torch.log(dist_an/(dist_an+dist_ap)) )/dist_an.size(0)
		# return loss

		# assert dist_an.size(0)==dist_ap.size(0)
		# y = torch.ones_like(dist_ap)
		# loss = self.margin_loss(dist_an, dist_ap, y)
		# prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)
		# return loss
		# loss = cal_triplet_from_an_ap(dist_ap,dist_an,self.margin,ensemble_weight)
		# return loss

class SoftTripletLoss(nn.Module):

	def __init__(self, margin=None, normalize_feature=True):
		super(SoftTripletLoss, self).__init__()
		self.margin = margin
		self.normalize_feature = normalize_feature

	def forward(self, emb1, emb2, label):
		if self.normalize_feature:
			# equal to cosine similarity
			emb1 = F.normalize(emb1)
			emb2 = F.normalize(emb2)

		mat_dist = euclidean_dist(emb1, emb1)
		assert mat_dist.size(0) == mat_dist.size(1)
		N = mat_dist.size(0)
		mat_sim = label.expand(N, N).eq(label.expand(N, N).t()).float()

		dist_ap, dist_an, ap_idx, an_idx = _batch_hard(mat_dist, mat_sim, indice=True)
		assert dist_an.size(0)==dist_ap.size(0)
		triple_dist = torch.stack((dist_ap, dist_an), dim=1)
		triple_dist = F.log_softmax(triple_dist, dim=1)
		if (self.margin is not None):
			loss = (- self.margin * triple_dist[:,0] - (1 - self.margin) * triple_dist[:,1]).mean()
			return loss

		mat_dist_ref = euclidean_dist(emb2, emb2)
		dist_ap_ref = torch.gather(mat_dist_ref, 1, ap_idx.view(N,1).expand(N,N))[:,0]
		dist_an_ref = torch.gather(mat_dist_ref, 1, an_idx.view(N,1).expand(N,N))[:,0]
		triple_dist_ref = torch.stack((dist_ap_ref, dist_an_ref), dim=1)
		triple_dist_ref = F.softmax(triple_dist_ref, dim=1).detach()

		loss = (- triple_dist_ref * triple_dist).mean(0).sum()
		return loss
