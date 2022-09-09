import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import pdb 
from sklearn.tree import DecisionTreeRegressor
from io import StringIO  
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import pydotplus

from sklearn.datasets import fetch_california_housing

def plot_tree(dtree, feature_names):
    """ helper function """
    dot_data = StringIO()
    export_graphviz(dtree, out_file=dot_data,
                    filled=True, rounded=True,
                    special_characters=True, feature_names=feature_names)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    print('exporting tree to dtree.png')
    graph.write_png('dtree.png')


class RegressionStump():
    
    def __init__(self):
        """ The state variables of a stump"""
        self.idx = None
        self.val = None
        self.left = None
        self.right = None
    
    def fit(self, data, targets):
        """ Fit a decision stump to data
        
        Find the best way to split the data, minimizing the least squares loss of the tree after the split 
    
        Args:
           data: np.array (n, d)  features
           targets: np.array (n, ) targets
    
        sets self.idx, self.val, self.left, self.right
        """
        # update these three
        self.idx = 0
        self.val = None
        self.left = None
        self.right = None
        ### YOUR CODE HERE
        num_points, num_features = data.shape
        print(num_points)
        best_score = None
        for feat_idx in range(num_features):
            feat = data[:, feat_idx]
            for split_idx in range(num_points):
                split_value = feat[split_idx]
                goes_left = feat < split_value
                goes_right = feat >= split_value
                if np.sum(goes_left)==0 or np.sum(goes_right)==0:
                    continue
                left_mean = np.sum(targets[goes_left])/np.sum(goes_left)
                right_mean = np.sum(targets[goes_right])/np.sum(goes_right)
                left_score = np.sum((targets[goes_left] - left_mean)**2)
                right_score = np.sum((targets[goes_right] - right_mean)**2)
                total_score = left_score + right_score
                if best_score == None or total_score < best_score:
                    best_score = total_score
                    self.idx = feat_idx
                    self.val = split_value
                    self.left = left_mean
                    self.right = right_mean
        ### END CODE

    def predict(self, X):
        """ Regression tree prediction algorithm

        Args
            X: np.array, shape n,d
        
        returns pred: np.array shape n,  model prediction on X
        """
        pred = None
        ### YOUR CODE HERE
        dat = X[:, self.idx]
        n = X.shape[0]        
        decision = (dat < self.val)
        pred = decision * self.left + (1-decision) * self.right
        ### END CODE
        return pred
    
    def score(self, X, y):
        """ Compute mean least squares loss of the model

        Args
            X: np.array, shape n,d
            y: np.array, shape n, 

        returns out: scalar - mean least squares loss.
        """
        out = None
        ### YOUR CODE HERE
        pred = self.predict(X)
        out = ((pred-y)**2).mean()
        ### END CODE
        return out
        


def main():
    """ Simple method testing """
    housing = fetch_california_housing()
    # split 80/20 train-test
    X_train, X_test, y_train, y_test = train_test_split(housing.data,
                                                        housing.target,
                                                        test_size=0.2)

    baseline_accuracy = np.mean((y_test-np.mean(y_train))**2)
    print('Least Squares Cost of learning mean of training data:', baseline_accuracy) 
    print('Lets see if we can do better with just one question')
    D = RegressionStump()
    D.fit(X_train, y_train)
    print('idx, val, left, right', D.idx, D.val, D.left, D.right)
    print('Feature name of idx', housing.feature_names[D.idx])
    print('Score of model', D.score(X_test, y_test))
    print('lets compare with sklearn decision tree')
    dc = DecisionTreeRegressor(max_depth=1)
    dc.fit(X_train, y_train)
    dc_score = ((dc.predict(X_test)-y_test)**2).mean()
    print('dc score', dc_score)
    print('feature names - for comparison', list(enumerate(housing.feature_names)))
    plot_tree(dc, housing.feature_names)

if __name__ == '__main__':
    main()