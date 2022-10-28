import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_gaussian_quantiles

class AdaBoostClassifier():

    def __init__(self, weak_learner, n_estimators=2):
        """ 

        Args:
          week_learner: weak learner object. Must support fit(self, x, y, weights), and predict(self, X) like sklearn classifiers. 
          We assume that the week learner predictions are in {-1, 1} as used in AdaBoost 

          n_estiamtors: int, number of estimators to construct
        """
        self.models = []
        self.alphas = []
        self.weak_learner = weak_learner
        self.n_estimators = n_estimators
    
    def fit(self, X, y):
        """ 
        AdaBoost Learning Algorithm 
        
        Args:
        X: numpy array shape (n,d) - the data (rows)
        y: numpy array shape (n,) all elements in {-1, 1} - the labels
            
        Computes and stores 
          - models: lists of size n_estimators of weak_learner
          - alphas: lists of size n_estimators of float

        Returns:
          scores: list of scores (accuracy) for each iteration of the algorithm (one value for each model considered)
          exp_losses: list of exponential losses for each iteration of the algorithm (one value for each model considered)

        to create a weak learners use 
        tmp = self.weak_learner()
        tmp.fit(X, y, p)
        """  
        w = np.ones(X.shape[0])/X.shape[0]
        scores = []
        exp_losses = []
        ### YOUR CODE HERE 
        for i in range(self.n_estimators):
            cur_score = self.score(X, y)
            scores.append(cur_score)
            exp_loss = self.exp_loss(X, y)
            exp_losses.append(exp_loss)
            pred = self.ensemble_output(X)
            w = np.exp(-y * pred)
            p = w/np.sum(w)
            wl = self.weak_learner()
            wl.fit(X, y, p)
            new_pred = wl.predict(X)
            correct_idx = (new_pred == y)
            wp = np.sum(w[correct_idx])
            wn = np.sum(w[~correct_idx])
            alphat = (0.5) * np.log(wp/wn)
            self.models.append(wl)
            self.alphas.append(alphat)
            
        scores.append(self.score(X, y))
        exp_loss = self.exp_loss(X, y)
        exp_losses.append(exp_loss)
        ### END CODE

        # remember to ensure that self.models and self.alphas are filled
        assert len(self.models) == self.n_estimators
        assert len(self.alphas) == self.n_estimators
        return scores, exp_losses
        

    def exp_loss(self, X, y):
        """ Compute Mean Exponential Loss of the data with the model (1/n sum_i exp(-y_i f(x_i)))

        Args:
        X: numpy array shape (n, d) - the data (rows)
        y: numpy array shape (n,) all elements in {-1, 1} - the labels

        Returns:
          loss: mean exponential loss
        """

        loss = None
        ### YOUR CODE here 1-3 lines
        pred = self.ensemble_output(X)
        loss = np.exp(-y * pred).mean()
        ### END CODE
        return loss
        

    def ensemble_output(self, X):
        """ Compute the output of the ensemble on the data (sum_i a_i h_i(x_i)) for all data points

        Args:
        X: numpy array shape (n, d) - the data (rows)
        
        Returns:
          pred: np.array (n, ) ensemble output on each input point in X (rows)
        """
        pred = None
        if len(self.models) == 0:
            return np.zeros(X.shape[0])
        ### YOUR CODE HERE 3-8 lines
        all_model_preds = [alpha * model.predict(X) for (alpha, model) in zip(self.alphas, self.models)]
        pred = np.sum(all_model_preds, axis=0)
        ### END CODE
        return pred
        
    def predict(self, X):
        """ predict function for classifier
        Args:
          X (numpy array,  shape (n, d)), - the data (rows)
        Returns
          pred (numpy array,  shape(n,)) values in {-1,+1}, prediction of model on data X - the labels
        """
        pred = None
        ### YOUR CODE Here 1-3 lines
        pred = np.sign(self.ensemble_output(X))
        ### END CODE 
        return pred

    def score(self, X, y):
        """ Return accuracy of model on data X with labels y ((1/n) (sum_i 1_[f(x_i) == y_i]))
        
        Args:
          X (numpy array shape n, d)
        returns
          score (float) classifier accuracy on data X with labels y
        """
        score = 0
        ### YOUR CODE HERE 1-3 lines
        pred = self.predict(X)
        indicator = pred==y
        score = indicator.mean()
        ### END CODE
        return score

        
def sklearn_test():
   """ AdaBoost test taken from sklearn
   https://scikit-learn.org/stable/auto_examples/ensemble/plot_adaboost_twoclass.html#sphx-glr-auto-examples-ensemble-plot-adaboost-twoclass-py"""

   # Construct dataset
   x1_samples = 200
   x2_samples = 300
   X1, y1 = make_gaussian_quantiles(cov=2.,
                                 n_samples=x1_samples, n_features=2,
                                 n_classes=2, random_state=1)
   X2, y2 = make_gaussian_quantiles(mean=(3, 3), cov=1.5,
                                 n_samples=x2_samples, n_features=2,
                                 n_classes=2, random_state=1)
   X = np.concatenate((X1, X2))
   y = np.concatenate((y1, - y2 + 1))
   y = 2*y-1
   
   # Create and fit an AdaBoosted decision tree
   my_learner = lambda: DecisionTreeClassifier(max_depth=1)
   bdt = AdaBoostClassifier(my_learner,
                            n_estimators=200)

   scores, exp_losses = bdt.fit(X, y)
   print('Final Accuracy', scores[-1])
   fig, ax = plt.subplots(1, 2, figsize=(12,10))
   ax[0].plot(exp_losses, 'b--', label='exp loss')
   ax[0].plot(1.0 - np.array(scores), 'm--', label='0-1 Loss')
   ax[0].legend(fontsize=15)
   ax[0].set_title('Loss Per Iteration for AdaBoost', fontsize=20)
   
   #plot_colors = "br"
   plot_step = 0.02
   #class_names = "AB"

   # Plot the decision boundaries
   x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
   y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
   xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                     np.arange(y_min, y_max, plot_step))

   Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
   Z = Z.reshape(xx.shape)
   cs = ax[1].contourf(xx, yy, Z, cmap=plt.cm.Paired)

   ax[1].scatter(X[:, 0], X[:, 1], c=y, s=20, cmap=plt.cm.Paired, edgecolor='k')
   ax[1].set_title('AdaBoost Decision Boundary', fontsize=20)
   plt.show()
    
if __name__=='__main__':
    sklearn_test()
