import numpy as np
from typing import List
from classifier import Classifier

class DecisionTree(Classifier):
	def __init__(self):
		self.clf_name = "DecisionTree"
		self.root_node = None

	def train(self, features: List[List[float]], labels: List[int]):
		# init.
		assert(len(features) > 0)
		self.feautre_dim = len(features[0])
		num_cls = np.max(labels)+1

		# build the tree
		self.root_node = TreeNode(features, labels, num_cls)
		if self.root_node.splittable:
			self.root_node.split()

		return
		
	def predict(self, features: List[List[float]]) -> List[int]:
		y_pred = []
		for feature in features:
			y_pred.append(self.root_node.predict(feature))
		return y_pred

	def print_tree(self, node=None, name='node 0', indent=''):
		if node is None:
			node = self.root_node
		print(name + '{')
		
		string = ''
		for idx_cls in range(node.num_cls):
			string += str(node.labels.count(idx_cls)) + ' '
		print(indent + ' num of sample / cls: ' + string)

		if node.splittable:
			print(indent + '  split by dim {:d}'.format(node.dim_split))
			for idx_child, child in enumerate(node.children):
				self.print_tree(node=child, name= '  '+name+'/'+str(idx_child), indent=indent+'  ')
		else:
			print(indent + '  cls', node.cls_max)
		print(indent+'}')


class TreeNode(object):
	def __init__(self, features: List[List[float]], labels: List[int], num_cls: int):
		self.features = features
		self.labels = labels
		self.children = []
		self.num_cls = num_cls
		count_max = 0
		for label in np.unique(labels):
			if self.labels.count(label) > count_max:
				count_max = labels.count(label)
				self.cls_max = label # majority of current node

		if len(np.unique(labels)) < 2:
			self.splittable = False
		else:
			self.splittable = True

		self.dim_split = None # the index of the feature to be split

		self.feature_uniq_split = None # the possible unique values of the feature to be split


	def split(self):
		def conditional_entropy(branches: List[List[int]]) -> float:
			'''
			branches: C x B array, 
					  C is the number of classes,
					  B is the number of branches
					  it stores the number of 
					  corresponding training samples 
					  e.g.
					              ○ ○ ○ ○
					              ● ● ● ●
					            ┏━━━━┻━━━━┓
				               ○ ○       ○ ○
				               ● ● ● ●
				               
				      branches = [[2,2], [4,0]]
			'''
			########################################################
			# TODO: compute the conditional entropy
			########################################################
			branches = np.array(branches)
			branch = np.sum(branches,axis=0)
			p = branches / branch
			ratio = branch / np.sum(branch)
			np.warnings.filterwarnings('ignore')
			cond_entropy = - np.sum(np.sum(np.nan_to_num(p * np.log2(p)),axis=0) * ratio)
			return cond_entropy
		
		features = np.array(self.features)
		labels = np.array(self.labels)
		unique_labels = np.unique(self.labels)

		entropies = []
		for idx_dim in range(len(self.features[0])):
		############################################################
		# TODO: compare each split using conditional entropy
		#       find the best split
		############################################################
			branches = np.unique(features[:,idx_dim])
			branch_matrix = np.zeros((unique_labels.size, branches.size))
			for b, branch in enumerate(branches):
				for c, label in enumerate(unique_labels):
					this_branch = features[:,idx_dim] == branch
					this_class = labels == label
					branch_matrix[c,b] = np.sum(np.logical_and(this_branch, this_class)).astype(int)
			entropies.append(conditional_entropy(branch_matrix.tolist()))
		self.dim_split = np.argmin(entropies)

		############################################################
		# TODO: split the node, add child nodes
		############################################################
		this_feature = features[:,self.dim_split]
		self.feature_uniq_split = np.unique(this_feature).tolist()
		children_features = np.delete(features,self.dim_split,1)
		for branch in self.feature_uniq_split:
			this_branch = this_feature == branch
			this_branch_feature = children_features[this_branch,:].tolist()
			this_branch_label = labels[this_branch].tolist()
			this_branch_num_cls = np.unique(labels[this_branch]).size
			child = TreeNode(this_branch_feature, this_branch_label, this_branch_num_cls)
			if len(this_branch_feature)==0:
				child.splittable = False
				child.cls_max = self.cls_max
			elif len(this_branch_feature[0])==0:
				child.splittable = False
			self.children.append(child)

		# split the child nodes
		for child in self.children:
			if child.splittable:
				child.split()

		return

	def predict(self, feature: List[int]) -> int:
		if self.splittable:
			idx_child = self.feature_uniq_split.index(feature[self.dim_split])
			return self.children[idx_child].predict(feature[:self.dim_split]+feature[(self.dim_split+1):])
		else:
			return self.cls_max



