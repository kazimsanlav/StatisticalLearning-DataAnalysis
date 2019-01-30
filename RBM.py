# %%
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import copy

np.random.seed(1)  # set the seed for reproducibility

# %%
tr_X = sio.loadmat('ps4q4.mat')['tr_X'].T
tr_y = sio.loadmat('ps4q4.mat')['tr_y']
ts_X = sio.loadmat('ps4q4.mat')['ts_X'].T
ts_y = sio.loadmat('ps4q4.mat')['ts_y']
# %%
index = np.random.permutation(np.arange(60000))[:10000]

tr_X = (tr_X[index, :])
tr_y = (tr_y[index])

# %%



def sigmoid(x): return(1/(1+np.exp(-x)))


class RBM(object):
    '''
    Restricted Boltzmann Machine (RBM) class
    '''

    def __init__(self, shape=(10, 784)):
        '''
        initialize the weights randomly
        '''
        self.shape = shape
        self.W = np.random.ranf(size=self.shape)
        self.b = np.random.ranf(size=(self.shape[1], 1))
        self.c = np.random.ranf(size=(self.shape[0], 1))

    def get_parameters(self):
        return {
            'W': self.W,
            'b': self.b,
            'c': self.c
        }

    def set_parameters(self, parameters):
        try:
            self.W = parameters['W']
            self.b = parameters['b']
            self.c = parameters['c']
            print('Parameters loaded succesfully')
        except:
            print('Error: check given parameters')

    def fit(self, data, epochs=20, batch_size=10, l_rate=0.05, cd_k=1):
        '''
        train the RBM using CD
        data is (n,784)
        batch is (batch_size,784)
        '''
        # data = np.random.shuffle(data)  # shuffle
        print('Training started...')
        for epoch in range(epochs):
            print('epoch -> {}/{}'.format(epoch+1, epochs))
            for i in range(0, data.shape[0], batch_size):

                batch = data[i:i+batch_size, :]  # create the batch

                gradient_W = np.zeros_like(self.W)
                gradient_b = np.zeros_like(self.b)
                gradient_c = np.zeros_like(self.c)

                for j in range(batch.shape[0]):

                    x_j = batch[j, :].reshape(-1)
                    x_sampled = self.sample(x_j, k=cd_k)
                    
                    h_xj = self.transform(x_j)
                    h_xsampled = self.transform(x_sampled)

                    # add up gradients
                    gradient_W += sigmoid(np.squeeze(self.c)+self.W@x_j)[:,np.newaxis]@x_j[np.newaxis] \
                                -  sigmoid(np.squeeze(self.c)+self.W@x_sampled)[:,np.newaxis]@x_sampled[np.newaxis]
                   
                    gradient_b += (x_j - x_sampled)[:,np.newaxis]
                   
                    gradient_c += sigmoid(np.squeeze(self.c)+self.W@x_j)[:,np.newaxis] \
                                -  sigmoid(np.squeeze(self.c)+self.W@x_sampled)[:,np.newaxis]

                    # gradient_W += h_xj@x_j.T - h_xsampled@x_sampled.T
                    # gradient_b += x_j - x_sampled
                    # gradient_c += h_xj - h_xsampled

                # update parameters after each batch
                self.W += l_rate*(gradient_W/batch_size)
                self.b += l_rate*(gradient_b/batch_size)
                self.c += l_rate*(gradient_c/batch_size)

            # to see if it is learning the weights
            # sample = self.sample(tr_X[1, :], k=cd_k)
            # self.show(tr_X[1, :])
            # self.show(sample)

    def sample(self, x_t, k=1):
        '''
        gibbs sampling, k steps, starting from x(t)
        '''
        # h_ = np.zeros_like(self.c)
        x_ = copy.deepcopy(x_t)

        for step in range(k):
            # get h using p(h|x)
            h_ = np.where(sigmoid(np.squeeze(self.c)+self.W@x_ ) > np.random.rand(self.c.size),1,0)
            
            # get x~ using p(x~|h)
            x_ = np.where(sigmoid(np.squeeze(self.b)+h_.T@self.W) > np.random.rand(self.b.size),1,0)

        # return final x^k
        return x_

    def transform(self, x):
        '''
        get latent(hidden) outputs given x 
        '''
        x_ = copy.deepcopy(x)
        # get h using p(h|x)
        h_ = np.where(np.squeeze(self.c)+self.W@x_ > np.random.rand(self.c.size),1,0)
        return h_

    def show(self, x):
        '''
        showing array as an image
        '''
        plt.imshow(x.reshape(28, 28), cmap='gray')
        plt.show()




