import numpy as np
from sklearn.metrics import accuracy_score
import data_loader
import decision_tree

###############
# Toy example #
###############
'''
Toy example

dim_1
 ┃
 ╋       ○
 ┃
 ╋   ×       ○
 ┃
 ╋       ×
 ┃
━╋━━━╋━━━╋━━━╋━ dim_0

Print the tree and check the result by yourself!
             
'''
# data
features, labels = data_loader.toy_data_3()
# build the tree
dTree = decision_tree.DecisionTree()
dTree.train(features, labels)
y_est_train = dTree.predict(features)
train_accu = accuracy_score(y_est_train, labels)
print('train_accu', train_accu)
# print
dTree.print_tree()