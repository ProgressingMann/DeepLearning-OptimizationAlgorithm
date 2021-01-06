#!/usr/bin/env python
# coding: utf-8

# # This notebook will include steps to perform complete gradient descend 

# ### Gradient Descend is an optimization algorithm which tunes the parameters of our neural network to minimize the loss function. 

# In[1]:


#There are many Loss functions, but we will use the logloss(also called sigmoid loss) function.
#L(yhat, y) = -(y*logyhat - (1-y)*log(1-yhat))    , where yhat is our predicted label and y is the actual label.


# ## One step Gradient Descend consists of 2 important parts:-
#     1.Forward Propogation
#     2.Backward Propogation
#     

# # 1) Forward propogation

# ## 1.1) Single Layer, Single Node
#     In this basic level of neural network, we bascially have only one node which inputs the data x and outputs a prediction
#     yhat.Also, consider only one training example x.

# To predict the label, we need 2 parameters, weights(w) and bias(b) and a linear equation:- z = (w.T)x + b. Here, z is the linear outcome.
# We then pass the value of z to an activation function(g(z)) which gives us a number between 0 and 1 through which we predict the label.
# So, the first step is to initialize our model's parameters, w and b.
# 

# In[2]:


import numpy as np
from testCases import *


# In[36]:


import matplotlib.pyplot as plt
import h5py
import scipy.io
import sklearn
import sklearn.datasets


# In[3]:


def initialize_parameters(n_x, n):
    """
    Arguments:
        x is an input vector of shape(n_x, 1).
        n is the number of hidden units inside the layer.
    
    Return:
        A dictionary named parameters containing the Weight and bias.
    """
    parameters = {}
    
    w = np.random.randn(n_x, n)
    b = np.random.randn(n, 1)
    
    parameters = {
        "w":w,
        "b":b
    }
    
    return parameters


# In[4]:


# # As usual, we will start by initializing parameters.
# def initialize_parameters_L(layers_dims):
#     """
#     Arguments:
#         Layers_dims is a vector containing the number of nodes in each layer(lth).
        
#     Returns:
#         Parameters, a dictionary containg our models parameters, weights and biases.
#     """
    
#     parameters = {}
    
#     np.random.seed(3)
    
#     for l in range(1, len(layers_dims)):
#         parameters['W'+str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1])*0.01
#         parameters['b'+str(l)] = np.zeros((layers_dims[l], 1))
        
#     return parameters


# In[5]:


def initialize_parameters_L(layers_dims):
    """
    Arguments:
        layers_dims is an array containing number of nodes in respective layer.
        
    Returns:
        parameters, a dictionary containing weights and biases for every layer.
    """
    
    L = len(layers_dims)
    parameters = {}
    
    for l in range(1, L):
        parameters["W"+str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2 / layers_dims[l-1])
        parameters["b"+str(l)] = np.zeros((layers_dims[l], 1))
        
    return parameters
    


# In[6]:


# xtemp = [3, 3, 1]
# initialize_parameters_L(xtemp)


# In[7]:


def sigmoid(z):
    """
        Arguments:
            Value of z to compute the activation value for z which is a.
            
        Returns:
            Activation value for z.
    """
    
    a = 1 / (1 + np.exp(-z))
    
    return a, z


# In[8]:


def relu(Z):
    """
    """
    
    return Z * (Z > 0), Z


# In[9]:


#a11 = np.random.randn(3, 2)
#sigmoid(a11)[0] * sigmoid(a11)[0]


# ## 1.2) Multiple Layers, Multiple nodes

#     In reality, neural networks tend to have many layers, input layer(n_x), hidden layers(n_h) and an output layer(n_y). Input layer is also considered as a 0th layer and hence is not counted in the overall number of layers.
#     If there are L total layers, there will be (L-1) hidden layers and the Lth layer will be the output layer of the neural network.
#     Each and every layer of a neural network consists of multiple nodes and each node consist of:-
#     1) Weights
#     2) Biases
#     3) Linear function(z)
#     4) Activation function(a)
# 
#     So, we will have to compute the forward propogation for each layer.
#     
#     

# ### Dimensions of layers and parameters.

# #### Layers:
#     Layers range from 1 to L.
#     If there are a total of L layers, there will be (L-1) hidden layers and each hidden layer have a certain number of nodes.
#     We have an input layer n_x of size(n_x, 1). It is also reffered as the 0th layer.
#     We keep an array n_h, where n_h[i] = number of nodes in layer[i+1](Since indexing starts from 0).
#     Similarly we have an output layer n_y which contains the number of outputs from the final layer.
#     Combination of hidden layers(n_h) and output layer(n_y) is stored in layers_dims.

# #### Weights:
#     If there are n nodes in first layer, each and every node in the first layer will have a weights vector of shape(n_x, 1).
#     Hence, we will have n such vectors of shape (n_x, 1).
#     We stack the weights vector columnwise(n_x, n) and then perform a transpose to them which gives us the matrix W1(1 for the
#     first layer, l for the lth layer) of shape (n, n_x). This matrix, W1, is the weights matrix for the first layer of our 
#     neural network.
#     Therefore, shape of a weights matrix for lth layer is :- Wl.shape = (layers_dims[l], layers_dim[l-1]).

# #### Biases:
#     For L total layers, and their respective number_of_nodes being stored in layers_dims, each layer will have a bias vector of 
#     shape (layers_dims[l], 1).

# #### Z:
#     We are supposed to calculate the linear function, z, for each and every node in the neural network.
#     So, we will compute z for each node in layer l, and then store all those z values of lth layer in a vector zl.
#     Therefore, shape of zl = (layers_dims[l], 1).
#     We need this z to calculate the activation a.
#     General Equation for z is:-
#     z = (Wl.T)a[l-1] + bl
#     That's the reason we consider input layer as 0th layer as inputs(n_x, 1) = a0(n_x, 1). a0 is input vector.

# #### A:
#     We apply the value of z to an activation function, which can be sigmoid, relu, tanh or any other function.
#     The value we get after applying the activation function to z will be given to the next next layers input.
#     Shape of al = (layers_dims[l], 1). It have to be same as of zl.

# ## Vectorization
#     To compute z and a across a training dataset of m instances, we might think of iterating through each instance m times.
#     But, this is computationally very expensive.
#     Instead of using a for loop, what we can do is, we can convert a vector zl and al, of shapes (layers_dims[l], 1), into 
#     matrix of shape(layers_dims[l], m).
#     By doing this, we can save a huge amount of time and computation will be reduced dramatically.
#     What we do is this:- 
#         Instead of feeding the input x of shape(n_x, 1) for m times, we will now input a matrix X of shape(n_x, m).
#         This will automatically change the dimensions of a and z.
#         We will now have matrices Zl and Al where each row in the matrix represents the nth node in lth layer and columns
#         represent the number(mth) of the training instance.

# In[10]:


# initialize_parameters_L([5, 4, 3])


# In[11]:


def linear_forward_propogation(A, W, b):
    """
    Arguments:
        A is an input vector with shape(layers_dims[l-1], m).
        W and b are parameters, the weights and biases.
    
    Returns:
        A linear computational output z.
    """
    Z = np.dot(W, A) + b
    cache = (A, W, b)
    
    return Z, cache


# In[12]:


# A, W, b = linear_forward_test_case()
# Z, linear_cache = linear_forward_propogation(A, W, b)
# print("Z = " + str(Z))
# print(linear_cache[0])


# In[13]:


def activation_forward_propogation(A_prev, W, b, activation='sigmoid'):
    """
    Argument:
        A_prev, the outputs from the previous layer of shape(layers_dims[l-1], m).
        W and b contains the weights and biases of layer l.
    Returns:    
        A, Activation matrix(Inputs to next layer/final output) of shape(layers_dims[l], 1).
        cache(A_prev, W, b) contains the matrices needed to compute Z and A for current layer(l).
        
    Linear_cache contains the parameters needed to compute Zl, which is in turn used to compute Al. 
    Activation_cache is Zl, the input to the activation function of that layer.
        
    """
    
    Z, linear_cache = linear_forward_propogation(A_prev, W, b)
    
    if activation=='sigmoid':
        A, activation_cache = sigmoid(Z)
    elif activation=='relu':
        A, activation_cache = relu(Z)
    
    cache = (linear_cache, activation_cache)
    
    return A, cache
    


# In[14]:


# A_prev, W, b = linear_activation_forward_test_case()

# A, linear_activation_cache = activation_forward_propogation(A_prev, W, b, activation = "sigmoid")
# print("With sigmoid: A = " + str(A))

# A, linear_activation_cache = activation_forward_propogation(A_prev, W, b, activation = "relu")
# print("With ReLU: A = " + str(A))


# In[15]:


#Next, Implementing Forward_Propogation through L layers of the network.
def forward_propogation_L(X, parameters):
    """
    Arguments:
        X, input dataset of shape(n_x, m).
        parameters, Weights and biases for each layer l.
    Returns:
        Matrix Z.
    """
    
    caches = []
    layers = len(parameters) // 2
    A = {}
    A_prev = X
    
    
    for l in range(1, layers):
        A, cache = activation_forward_propogation(A_prev, parameters["W"+str(l)], parameters["b"+str(l)], activation='relu')
        caches.append(cache)
        A_prev = A
    
    AL, cache = activation_forward_propogation(A_prev, parameters["W"+str(layers)], parameters["b"+str(layers)], activation='sigmoid')
    caches.append(cache)
    
    return AL, caches


# In[16]:


# X, parameters = L_model_forward_test_case_2hidden()
# AL, caches = forward_propogation_L(X, parameters)
# print("AL = " + str(AL))
# print("Length of caches list = " + str(len(caches)))
# print(caches)


# # 2) Computing the cost of our model

# In[17]:


def compute_cost(AL, Y, cost='logloss'):
    """
    Arguments:
        AL, the output from the last layer of our model. They are our predicted labels. It has shape(1, m). 
        Y, the actual labels(True labels) of dataset. It also has shape(1, m).
    
    Returns:
        Loss of our model.
    """
    
    m = AL.shape[1]
    cost = 0
    
    cost = -np.sum(Y*np.log(AL) + (1-Y)*np.log(1-AL)) / m
    return cost
    


# In[18]:


# Y, AL = compute_cost_test_case()

# print("cost = " + str(compute_cost(AL, Y)))


# # 3) BackPropogation

# In[19]:


def relu_backward(dA, cache):
    """
    Arguments:
        dA, the vector on which we have to perform derivative of relu.
        Z, a vector which is stored in activation_cache to compute A.
        
    Returns:
        dZ, the gradient of Z[l].
    """
    Z = cache
    dZ = np.array(dA, copy=True)
    
    dZ[Z < 0] = 0
    
    return dZ
    


# In[20]:


#np.random.seed(3)
#z11 = np.random.randn(3, 2)
#print(z11)
#np.multiply(z11 > 0, z11)


# In[21]:


def sigmoid_backward(dA, activation_cache):
    """
    Arguments:
        dA, the vector on which we have to perform derivative of relu.
        Z, a vector which is stored in activation_cache to compute A.
        
    Returns:
        dZ, the gradient of Z[l].
    """
    Z = activation_cache
    
    s = 1/(1 + np.exp(-Z))
    
    dZ = dA * s * (1 - s)
    
    return dZ
    #return np.multiply(dA, sigmoid(Z)[0]*(1-sigmoid(Z)[0]))
    


# In[22]:


def linear_backward(dZ, cache):
    """
    Arguments:
        dZ, a vector in lth layer of shape(layers_dims[l], 1).
        cache contains the weights[l], biases[l] and the input to this layer, A_prev, which we use to in forward_propogation 
        to compute Zl.
        
    Returns:
        grad(Gradients), a dictionary containing the derivatives of caches.
    """
    
    grads = {}
    A_prev, W, b = cache
    m = A_prev.shape[1]
    
    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db


# In[23]:


# # Set up some test inputs
# dZ, linear_cache = linear_backward_test_case()

# dA_prev, dW, db = linear_backward(dZ, linear_cache)
# print ("dA_prev = "+ str(dA_prev))
# print ("dW = " + str(dW))
# print ("db = " + str(db))


# In[24]:


def linear_activation_backward(dA, cache, activation):
    """
    Arguments:
        dA, the activation values of current layer[l].
        cache contains the weights, biases of current layer[l] and activation values(A_prev) of previous layer[l-1].
        activation, the type of activation function used at layer[l].
    Returns:
        grads(Gradients), a dictionary containing derivatives of caches, i.e. weights, biases and dA_prev.
    """
    
    linear_cache, activation_cache = cache
    
    if activation=='relu':
        dZ = relu_backward(dA, activation_cache)
    elif activation=='sigmoid':
        dZ = sigmoid_backward(dA, activation_cache)
    
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db


# In[25]:


#dAL, linear_activation_cache = linear_activation_backward_test_case()

#dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = "sigmoid")
#print ("sigmoid:")
#print ("dA_prev = "+ str(dA_prev))
#print ("dW = " + str(dW))
#print ("db = " + str(db) + "\n")

#dA_prev, dW, db = linear_activation_backward(dAL, linear_activation_cache, activation = "relu")
#print ("relu:")
#print ("dA_prev = "+ str(dA_prev))
#print ("dW = " + str(dW))
#print ("db = " + str(db))


# In[26]:


def backward_propogation_L(AL, Y, caches):
    """
    Arguments:
        AL is the output from our final layer L. 
        caches contains linear_cache, the inputs to current layer, weights and biases of current layer and activation_cache,
        which has values of Z of current layer.
        
    Returns:
        grads(Gradient), a dictionary containing derivatives/gradients of caches, i.e. dA, dW, db.
    """
    
    grads = {}
    dAL = np.divide(AL-Y, AL*(1-AL))
    m = AL.shape[1]
    L = len(caches)
    
    grads["dA"+str(L-1)], grads["dW"+str(L)], grads["db"+str(L)] = linear_activation_backward(dAL, caches[L-1], activation='sigmoid')
    
    
    for l in range(L-1, 0, -1):
        grads["dA"+str(l-1)], grads["dW"+str(l)], grads["db"+str(l)]  = linear_activation_backward(grads["dA"+str(l)], caches[l-1], activation='relu')
        
    return grads
        
        
    


# In[27]:


# AL, Y_assess, caches = L_model_backward_test_case()
# grads = backward_propogation_L(AL, Y_assess, caches)
# print_grads(grads)
# len(caches)


# In[ ]:





# In[28]:


def update_parameters(parameters, grads, learning_rate=0.01):
    """
    Arguments:
        Parameters contains the weights and biases for our model.
        grads contains the gradients which tells us in which direction should our parameters move.
        learning_rate controls the speed of our learning algorithm, or we can say it tells us how quickly we want to 
        minimize the loss function.
    """
    
    layers = len(parameters) // 2
    
    for l in range(1, layers+1):
        
        parameters["W"+str(l)] = parameters["W"+str(l)] - learning_rate*grads["dW"+str(l)]
        parameters["b"+str(l)] = parameters["b"+str(l)] - learning_rate*grads["db"+str(l)]
    
    return parameters


# In[29]:


# parameters, grads = update_parameters_test_case()
# parameters = update_parameters(parameters, grads, 0.01)

# print ("W1 = "+ str(parameters["W1"]))
# print ("b1 = "+ str(parameters["b1"]))
# print ("W2 = "+ str(parameters["W2"]))
# print ("b2 = "+ str(parameters["b2"]))


# In[ ]:





# In[30]:


def load_params_and_grads(seed=1):
    np.random.seed(seed)
    W1 = np.random.randn(2,3)
    b1 = np.random.randn(2,1)
    W2 = np.random.randn(3,3)
    b2 = np.random.randn(3,1)

    dW1 = np.random.randn(2,3)
    db1 = np.random.randn(2,1)
    dW2 = np.random.randn(3,3)
    db2 = np.random.randn(3,1)
    
    return W1, b1, W2, b2, dW1, db1, dW2, db2


# In[31]:


from sklearn import datasets
from matplotlib import pyplot as plt
def load_dataset():
    np.random.seed(3)
    train_X, train_Y = datasets.make_moons(n_samples=300, noise=.2) #300 #0.2 
    # Visualize the data
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    
    return train_X, train_Y


# In[104]:


def predict(X, y, parameters):
    """
    This function is used to predict the results of a  n-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    p = np.zeros((1,m), dtype = np.int)
    
    # Forward propagation
    AL, caches = forward_propogation_L(X, parameters)
    
    # convert probas to 0/1 predictions
    for i in range(0, AL.shape[1]):
        if AL[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    # print results

    #print ("predictions: " + str(p[0,:]))
    #print ("true labels: " + str(y[0,:]))
    print("Accuracy: "  + str(np.mean((p[0,:] == y[0,:]))))
    
    return p


# In[37]:


def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()


# In[38]:


def predict_dec(parameters, X):
    """
    Used for plotting decision boundary.
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (m, K)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Predict using forward propagation and a classification threshold of 0.5
    AL, cache = forward_propogation_L(X, parameters)
    predictions = (AL > 0.5)
    return predictions


# In[ ]:




