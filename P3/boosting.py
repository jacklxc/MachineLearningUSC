import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs      # set of weak classifiers to be considered
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		'''
		Inputs:
		- features: the features of all test examples
   
		Returns:
		- the prediction (-1 or +1) for each example (in a list)
		'''
		########################################################
		# TODO: implement "predict"
		########################################################
		total = 0
		for beta, clf in zip(self.betas, self.clfs_picked):
			total += beta * np.array(clf.predict(features))
		pred = []
		for element in total:
			if element > 0:
				pred.append(1)
			else:
				pred.append(-1)
		return pred
		

class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		'''
		Inputs:
		- features: the features of all examples
		- labels: the label of all examples
   
		Require:
		- store what you learn in self.clfs_picked and self.betas
		'''
		############################################################
		# TODO: implement "train"
		############################################################
		N = len(features)
		D = np.ones(N) / N
		y = np.array(labels)
		clfs = list(self.clfs)
		for t in range(self.T):
			min_index = np.argmin([np.sum(D[np.array(clf.predict(features)) != y]) for clf in clfs])
			h_t = clfs[min_index]
			self.clfs_picked.append(h_t)
			pred = h_t.predict(features)
			eps = np.sum(D[pred != y])
			beta = np.log((1-eps)/eps) / 2
			self.betas.append(beta)
			coef = np.ones(D.shape)
			coef[pred==y] = np.exp(-beta)
			coef[pred!=y] = np.exp(beta)
			D = D * coef
			D = D / np.sum(D)
		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)



	