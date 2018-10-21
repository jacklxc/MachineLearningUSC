import numpy as np
from sklearn.metrics import accuracy_score
import json

import data_loader
import decision_tree

#import matplotlib.pyplot as plt

# load data
X_train, X_test, y_train, y_test = data_loader.discrete_2D_iris_dataset()
"""
plt.figure()
X_test_np = np.array(X_test)
y_test_np = np.array(y_test)
plt.scatter(X_test_np[y_test_np==0,0], X_test_np[y_test_np==0,1],color="blue")
plt.scatter(X_test_np[y_test_np==1,0], X_test_np[y_test_np==1,1],color="red")
plt.scatter(X_test_np[y_test_np==2,0], X_test_np[y_test_np==2,1],color="yellow")
plt.show()
"""
# set classifier
dTree = decision_tree.DecisionTree()

# training
dTree.train(X_train, y_train)
y_est_train = dTree.predict(X_train)
train_accu = accuracy_score(y_est_train, y_train)
print('train_accu', train_accu)

# testing
y_est_test = dTree.predict(X_test)
test_accu = accuracy_score(y_est_test, y_test)
print('test_accu', test_accu)



# print
dTree.print_tree()

# save
json.dump({'train_accu': train_accu, 'test_accu': test_accu},
			open('decision_tree.json', 'w'))