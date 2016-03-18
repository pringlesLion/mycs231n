import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ConvNet(object):
  """
  A three-layer convolutional network with the following architecture:
  
  {conv -[spatial norm]- relu - 2x2 max pool} x N - {affine - [batch norm] - relu - [dropout]} x M - affine - softmax
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, input_dim=(3, 32, 32), num_filters=[32], filter_size=7,
               hidden_dims=[100], num_classes=10, weight_scale=1e-3, use_batchnorm=False, 
               dropout=0, reg=0.0, dtype=np.float32, seed=None):
    """
    Initialize a new network.
    
    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: A list of integers giving the number of filters to use in each convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dims: A list of integers giving the size of each fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - use_batchnorm:Whether or not the network should use batch normalization
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.use_batchnorm = use_batchnorm
    self.use_dropout = dropout > 0
    self.reg = reg
    self.num_linear_layers = 1 + len(hidden_dims)
    self.num_conv_layers = 1 + len(num_filters)
    self.num_layers = self.num_linear_layers + self.num_conv_layers - 1
    self.filter_size = filter_size
    self.conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}
    self.pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}
    self.dtype = dtype
    self.params = {}
    
    ############################################################################
    # Initialize weights and biases for the three-layer convolutional          #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    input_channel = C
    P = self.conv_param['pad']
    stride = self.conv_param['stride']
    
    
    for i in xrange(self.num_conv_layers-1):
        self.params['W'+str(i+1)] = weight_scale * np.random.randn(num_filters[i], input_channel, filter_size, filter_size)
        self.params['b'+str(i+1)] = np.zeros(num_filters[i])
        if self.use_batchnorm:
            self.params['gamma' + str(i+1)] = np.ones(num_filters[i])
            self.params['beta' + str(i+1)] = np.zeros(num_filters[i])
        input_channel = num_filters[i]
        #output height after conv layer 
        H = (H + 2 * P - filter_size) / stride + 1 
        W = (W + 2 * P - filter_size) / stride + 1 
        #output height after max pool layer
        H = (H - 2) / 2 + 1 
        W = (W - 2) / 2 + 1
    
    l_input_dim = input_channel * H * W
    for i in xrange(self.num_linear_layers-1):
        self.params['W' + str(self.num_conv_layers + i)] = weight_scale * np.random.randn(l_input_dim, hidden_dims[i])
        self.params['b' + str(self.num_conv_layers + i)] = np.zeros(hidden_dims[i])
        if self.use_batchnorm:
            self.params['gamma' + str(self.num_conv_layers + i)] = np.ones(hidden_dims[i])
            self.params['beta' + str(self.num_conv_layers + i)] = np.zeros(hidden_dims[i])
        l_input_dim = hidden_dims[i]
        
    self.bn_params = []
    if self.use_batchnorm:
        self.bn_params = [{'mode': 'train'} for i in xrange(self.num_layers - 1)]   
    
    self.dropout_param = {}
    if self.use_dropout:
      self.dropout_param = {'mode': 'train', 'p': dropout}
      if seed is not None:
        self.dropout_param['seed'] = seed
    
    self.params['W' + str(self.num_layers)] = weight_scale * np.random.randn(hidden_dims[-1], num_classes)
    self.params['b' + str(self.num_layers)] = np.zeros(num_classes)
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)
     
 
  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.
    
    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    # pass conv_param to the forward pass for the convolutional layer
    conv_param = self.conv_param
    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = self.pool_param
    # pass dropout_param to the forward pass for the dropout layer
    dropout_param = self.dropout_param

    scores = None
    ############################################################################
    # Implement the forward pass for the three-layer convolutional net,        #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    caches = [] 
    conv = X
    for i in xrange(self.num_conv_layers-1):
        Wi = self.params['W' + str(i+1)]
        bi = self.params['b' + str(i+1)]
        if self.use_batchnorm:
            gamma = self.params['gamma' + str(i+1)]
            beta = self.params['beta' + str(i+1)]
            bn_param = self.bn_params[i]
            conv, cache = conv_bn_relu_pool_forward(conv, Wi, bi, conv_param, pool_param, gamma, beta, bn_param)
        else:
            conv, cache = conv_relu_pool_forward(conv, Wi, bi, conv_param, pool_param)
        caches.append(cache)
    
    N, F, H, W = conv.shape 
    output = conv.reshape(X.shape[0], -1)
    
    for i in xrange(self.num_linear_layers-1):
        Wi = self.params['W' + str(self.num_conv_layers + i)]
        bi = self.params['b' + str(self.num_conv_layers + i)]
        if self.use_batchnorm:
            gamma = self.params['gamma' + str(self.num_conv_layers + i)]
            beta = self.params['beta' + str(self.num_conv_layers + i)]
            bn_param = self.bn_params[self.num_conv_layers + i -1]
            if self.use_dropout:
                output, cache = affine_bn_relu_dropout_forward(output, Wi, bi, gamma, beta, bn_param, dropout_param)
            else:
                output, cache = affine_bn_relu_forward(output, Wi, bi, gamma, beta, bn_param)
        else:
            if self.use_dropout:
                output, cache = affine_relu_dropout_forward(output, Wi, bi, dropout_param)
            else:
                output, cache = affine_relu_forward(output, Wi, bi)
        caches.append(cache)
    
    scores, cache_score = affine_forward(output, self.params['W' + str(self.num_layers)], self.params['b' + str(self.num_layers)])
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # Implement the backward pass for the three-layer convolutional net,       #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss,dscores = softmax_loss(scores, y)
    dx, grads['W' + str(self.num_layers)],  grads['b' + str(self.num_layers)] = affine_backward(dscores, cache_score) 
    for i in reversed(xrange(self.num_linear_layers-1)):
        if self.use_batchnorm:
            if self.use_dropout:
                dx, dw, db, dgamma, dbeta = affine_bn_relu_dropout_backward(dx, caches[self.num_conv_layers + i -1])
            else:
                dx, dw, db, dgamma, dbeta = affine_bn_relu_backward(dx, caches[self.num_conv_layers + i -1])
            grads['gamma' + str(self.num_conv_layers + i)] = dgamma
            grads['beta' + str(self.num_conv_layers + i)] = dbeta
        else:
            if self.use_dropout:
                dx, dw, db = affine_relu_dropout_backward(dx, caches[self.num_conv_layers + i -1])
            else:
                dx, dw, db = affine_relu_backward(dx, caches[self.num_conv_layers + i -1])
        grads['W' + str(self.num_conv_layers + i)] = dw
        grads['b' + str(self.num_conv_layers + i)] = db
        
    dx = dx.reshape((N,F,H,W))    
    for i in reversed(xrange(self.num_conv_layers-1)):
        if self.use_batchnorm:
            dx, dw, db, dgamma, dbeta = conv_bn_relu_pool_backward(dx, caches[i])
            grads['gamma' + str(i + 1)] = dgamma
            grads['beta' + str(i + 1)] = dbeta
        else:
            dx, dw, db = conv_relu_pool_backward(dx, caches[i])    
        grads['W' + str(i+1)] = dw
        grads['b' + str(i+1)] = db
   
    for i in xrange(self.num_layers - 1):
        W = self.params['W' + str(i+1)]
        loss += 0.5 * self.reg * np.sum(W**2) 
        grads['W' + str(i+1)] += self.reg * W
        
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
