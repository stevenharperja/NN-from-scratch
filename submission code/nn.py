# Place your EWU ID and Name here. Steven Harper 953052

### Delete every `pass` statement below and add in your own code. 



# Implementation of the forwardfeed neural network using stachastic gradient descent via backpropagation
# Support parallel/batch mode: process every (mini)batch as a whole in one forward-feed/backtracking round trip. 



import numpy as np
import math

from conda.core.initialize import make_initialize_plan

#import math_util as mu
#import nn_layer

#stuff to import code_misc
import sys
sys.path.append('..')
from code_misc.utils import MyUtils
import numpy as np
from code_NN.nn_layer import NeuralLayer
import code_NN.math_util as mu
import code_NN.nn_layer



###OUTPUT YOUR WS AS A SAVE THING FOR MICHAEL SO HE CAN JUST LOAD IN THE Ws

class NeuralNetwork:
    
    def __init__(self):
        self.layers = []     # the list of L+1 layers, including the input layer. 
        self.L = -1          # Number of layers, excluding the input layer. 
                             # Initting it as -1 is to exclude the input layer in L.
    def _prep_x(X):
        #add bias
        X = np.insert(X, 0, 1,axis=1)
        #normaliz X to be [-1,1] if tanh func, or [0,1] if logistic func
        #TODO
        return X
    
    def add_layer(self, d = 1, act = 'tanh'):
        ''' The newly added layer is always added AFTER all existing layers.
            The firstly added layer is the input layer.
            The most recently added layer is the output layer. 
            
            d: the number of nodes, excluding the bias node, which will always be added by the program. 
            act: the choice of activation function. The input layer will never use an activation function even if it is given. 
            
            So far, the set of supported activation functions are (new functions can be easily added into `math_util.py`): 
            - 'tanh': the tanh function
            - 'logis': the logistic function
            - 'iden': the identity function
            - 'relu': the ReLU function
        '''
        self.layers.append(NeuralLayer(d, act))
        self.L += 1
    

    def _init_weights(self):
        ''' Initialize every layer's edge weights with random numbers from [-1/sqrt(d),1/sqrt(d)], 
            where d is the number of nonbias node of the layer
        '''
        
        for l in range(1,len(self.layers)):
            w_shape = (self.layers[l-1].d + 1, self.layers[l].d)
            ##make the values be [-1/sqrt(d),1/sqrt(d)]
            self.layers[l].W = (np.random.rand(w_shape[0],w_shape[1])-0.5) * (2/math.sqrt(self.layers[l].d))
    
    def fit(self, X, Y, eta = 0.01, iterations = 1000, SGD = True, mini_batch_size = 1):
        ''' Find the fitting weight matrices for every hidden layer and the output layer. Save them in the layers.
          
            X: n x d matrix of samples, where d >= 1 is the number of features in each training sample
            Y: n x k vector of lables, where k >= 1 is the number of classes in the multi-class classification
            eta: the learning rate used in gradient descent
            iterations: the maximum iterations used in gradient descent
            SGD: True - use SGD; False: use batch GD
            mini_batch_size: the size of each mini batch size, if SGD is True.  
        '''
        self._init_weights()  # initialize the edge weights matrices with random numbers.

        
        
        # I will leave you to decide how you want to organize the rest of the code, but below is what I used and recommend. Decompose them into private components/functions.
        
        #last layer's index, L
        L = len(self.layers) - 1
        
        ## prep the data: add bias column; randomly shuffle data training set.
        self._init_weights()
        X = np.insert(X,0,1,axis=1)
        X,Y = NeuralNetwork._shuffle(X,Y)
        MyUtils.normalize_neg1_pos1(X)
        N = X.shape[0]

        #just set minibatch size to the full batch and run it just the same
        if(not SGD):
            mini_batch_size = X.shape[0]
        
        #loop stuff
        last_i = math.ceil(float(N)/float(mini_batch_size))
        i=0
        #### get a minibatch and use it for:
        while iterations > 0:
            start = i * mini_batch_size
            end = (i+1) * mini_batch_size
            if(end >= N):
                end = N
            minibatch_X = X[start:end]
            minibatch_Y = Y[start:end]
            
            self._gradient_descent(minibatch_X,minibatch_Y,eta)
            
            
            iterations -= 1
            i += 1
            if i >= last_i:
                i=0
            
            

    def _shuffle(X, Y):
        """
        n, d = np.shape(X)
        R = np.concatenate((X, Y), 1)
        np.random.shuffle(R)
        return np.split(R, d, axis=1)
        """
        assert len(X) == len(Y)
        p = np.random.permutation(len(X))
        return X[p], Y[p]

    def _gradient_descent(self,X,Y,eta):
        N = X.shape[0]
        ######### forward feeding
        self._ff(X)
        ######### calculate the error of this batch if you want to track/observe the error trend for viewing purpose.
        #E = self._curr_err(Y, N)
        ###calculate the final layer first to start the back propogation
        self._calc_last_layer(Y,N)
        ######### back propagation to calculate the gradients of all the weights
        self._back_propogate(N)
        ######### use the gradients to update all the weight matrices.
        self._update_weights(eta)
        

    def _calc_last_layer(self,Y,N):
        # last layer's index, L
        L = len(self.layers) - 1
        self.layers[L].Delta = 2 * (self.layers[L].X - Y) * self.layers[L].act_de(self.layers[L].S)
        self.layers[L].G = np.einsum('ij,ik->jk', self.layers[L - 1].X, self.layers[L].Delta) * 1 / N

    def _ff(self,in_X):
        #commenting this out because assuming the sample already has bias layer.
        #in_X = np.concatenate(np.ones(X.shape()[0]),X,axis=1)

        # last layer's index, L
        L = len(self.layers) - 1
        
        self.layers[0].X = in_X
        for l in range(1,len(self.layers)):
            #S = X with W
            self.layers[l].S = self.layers[l-1].X @ self.layers[l].W
            #X = signal(S) with bias column attached
            self.layers[l].X = np.insert(self.layers[l].act(self.layers[l].S) ,0,1,axis=1) #could i use the bias column without having to allocate memory every step? this seems inefficient
        
        self.layers[L].X = self.layers[L].X[:,1:]
        return self.layers[L].X
    
    def _curr_err(self,Y,N):
        # last layer's index, L
        L = len(self.layers) - 1
        
        #whats our mean squared error
        E = np.sum(np.square(self.layers[L].X[:, 1:] - Y))*(1/N)
        
    def _back_propogate(self,N):
        # last layer's index, L
        L = len(self.layers) - 1
        for l in reversed(range(1,L)):
            self.layers[l].Delta = self.layers[l].act_de(self.layers[l].S) * ( self.layers[l+1].Delta @ self.layers[l+1].W[1:,:].T )
            self.layers[l].G = np.einsum('ij,ik->jk', self.layers[l - 1].X, self.layers[l].Delta) / N
    def _update_weights(self,eta):
        # last layer's index, L
        L = len(self.layers) - 1
        for l in range(1,L+1):
            self.layers[l].W = self.layers[l].W - eta * self.layers[l].G
    def predict(self, X):
        ''' X: n x d matrix, the sample batch, excluding the bias feature 1 column.
            
            return: n x 1 matrix, n is the number of samples, every row is the predicted class id.
         '''
        #predict a batch of samples
        
        #add bias column, etc
        X = np.insert(X,0,1,axis=1)
        MyUtils.normalize_neg1_pos1(X)
        #output classification weights for each class from each sample
        class_weights = self._ff(X)
        #choose the most likely classification and return those as the representative 'y' for each sample.
        output = np.argmax(class_weights, axis=1)
        print(output)
        return output
    
    
    def error(self, X, Y):
        ''' X: n x d matrix, the sample batch, excluding the bias feature 1 column. 
               n is the number of samples. 
               d is the number of (non-bias) features of each sample. 
            Y: n x k matrix, the labels of the input n samples. Each row is the label of one sample, 
               where only one entry is 1 and the rest are all 0. 
               Y[i,j]=1 indicates the ith sample belongs to class j.
               k is the number of classes. 
            
            return: the percentage of misclassfied samples
        '''
        #set n ???
        n,d = X.shape
        #compare prediction and true values (Y)
        pred = self.predict(X)
        Y_indices = np.argmax(Y, axis=1)
        correct = np.sum(pred == Y_indices)
        #count up how many are wrong
        wrong = n - correct
        #divide that number by N and return as NOT a percent, but a decimal
        misclassified = (float(wrong)/float(n)) #* 100.0
        return misclassified

