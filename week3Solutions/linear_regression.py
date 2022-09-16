import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor

from sklearn.datasets import fetch_california_housing

import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
    getattr(ssl, '_create_unverified_context', None)): 
    ssl._create_default_https_context = ssl._create_unverified_context


class LinearRegressor():

    def __init__(self):
        self.w = None
        
    def hardcode_bias(self, X):
        """
        Append or prepend a hardcoded feature of value '1' to represent the bias.
        
        Args:
        X: numpy array shape (n,d)
        
        returns
        newX: copy of X, but with an extra feature hard-coded to 1, shape (n,d+1)
        
        Hint: np.concatenate may be useful
        """
        newX = X.copy()
        ### YOUR CODE HERE 1-3 lines
        n = X.shape[0]
        ones = np.ones((n,1))
        newX = np.concatenate((ones,newX),axis=1)
        ### END CODE
        return newX
    
    def fit(self, X, y):
        """ 
        Linear Regression Learning Algorithm
        
        For this we compute the parameter vector         
        wlin = argmin ( sum_i (w^T x_i -y_i)^2 )    
        The pseudo-inverse operator pinv in numpy.linalg package may be useful, i.e. np.linalg.pinv

        Args:
        X: numpy array shape (n,d)
        y: numpy array shape (n,)
            
        Computes and stores w: numpy array shape (d,) the best weight vector w to linearly approximate the target from the features.

        """  
        w = np.zeros(X.shape[1]+1)
        newX = self.hardcode_bias(X)
        ### YOUR CODE HERE 1-3 lines
        w = np.linalg.pinv(newX) @ y
        ### END CODE
        self.w =  w

    def predict(self, X):
        """ predict function for classifier
        Args:
          X (numpy array,  shape (n,d))
        Returns
          pred (numpy array,  shape(n,))
        """
        pred = None
        newX = self.hardcode_bias(X)
        ### YOUR CODE HERE 1-2 lines
        pred = newX @ self.w
        ### END CODE
        return pred

    def score(self, X, y):
        """ Return mean squared loss of model on data X with labels y
        
        Args:
          X (numpy array shape n, d)
        returns
          score (float) mean squared loss on data X with labels y
        """
        score = 0 
        ### YOUR CODE HERE 1-3 lines
        score = np.mean((self.predict(X) - y)**2)
        ### END CODE
        return score
        


def main():
    """ Simple method testing """
    housing = fetch_california_housing()
    # split 80/20 train-test
    X_train, X_test, y_train, y_test = train_test_split(housing.data,
                                                        housing.target,
                                                        test_size=0.2)


    baseline_accuracy = np.mean((y_test-np.mean(y_train))**2)
    print('Least Squares Cost of learning mean of training data:', baseline_accuracy) 
    print('Let\'s see if we can do better with linear regression')
    D = LinearRegressor()
    D.fit(X_train, y_train)
    print('Score of linear regression', D.score(X_test, y_test))
    
    print('Let\'s compare with sklearn decision stump')
    dc = DecisionTreeRegressor(max_depth=1)
    dc.fit(X_train, y_train)
    dc_score = ((dc.predict(X_test)-y_test)**2).mean()
    print('dc score', dc_score)
    print('feature names', list(enumerate(housing.feature_names)))

if __name__ == '__main__':
    main()