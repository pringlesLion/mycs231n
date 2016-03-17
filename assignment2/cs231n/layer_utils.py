from cs231n.layers import *
from cs231n.fast_layers import *


def affine_relu_forward(x, w, b):
  """
  Convenience layer that perorms an affine transform followed by a ReLU

  Inputs:
  - x: Input to the affine layer
  - w, b: Weights for the affine layer

  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, fc_cache = affine_forward(x, w, b)
  out, relu_cache = relu_forward(a)
  cache = (fc_cache, relu_cache)
  return out, cache


def affine_relu_backward(dout, cache):
  """
  Backward pass for the affine-relu convenience layer
  """
  fc_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = affine_backward(da, fc_cache)
  return dx, dw, db


pass


def conv_relu_forward(x, w, b, conv_param):
  """
  A convenience layer that performs a convolution followed by a ReLU.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  
  Returns a tuple of:
  - out: Output from the ReLU
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  out, relu_cache = relu_forward(a)
  cache = (conv_cache, relu_cache)
  return out, cache


def conv_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu convenience layer.
  """
  conv_cache, relu_cache = cache
  da = relu_backward(dout, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db


def conv_relu_pool_forward(x, w, b, conv_param, pool_param):
  """
  Convenience layer that performs a convolution, a ReLU, and a pool.

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer

  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  s, relu_cache = relu_forward(a)
  out, pool_cache = max_pool_forward_fast(s, pool_param)
  cache = (conv_cache, relu_cache, pool_cache)
  return out, cache


def conv_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, relu_cache, pool_cache = cache
  ds = max_pool_backward_fast(dout, pool_cache)
  da = relu_backward(ds, relu_cache)
  dx, dw, db = conv_backward_fast(da, conv_cache)
  return dx, dw, db

def conv_bn_relu_forward(x, w, b, conv_param, gamma, beta, bn_param):
  """
  Convenience layer that performs a convolution, a BN, and a ReLU

  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - bn_param: Parameters for the batch normalization layer

  Returns a tuple of:
  - out: Output from the relu layer
  - cache: Object to give to the backward pass
  """ 
  a, conv_cache = conv_forward_fast(x, w, b, conv_param)
  b, bn_cache = spatial_batchnorm_forward(x, gamma, beta, bn_param)
  out, relu_cache = relu_forward(b)
  cache = (conv_cache, bn_cache, relu_cache)
  return out, cache
  
def conv_bn_relu_backward(dout, cache):
  """
  Backward pass for the conv-relu-pool convenience layer
  """
  conv_cache, bn_cache, relu_cache = cache
  drelu = relu_backward(dout, relu_cache)
  dbn, dgamma, dbeta = spatial_batchnorm_backward(drelu, bn_cache)
  dx, dw, db = conv_backward_fast(dbn, conv_cache)
  return dx, dw, db, dgamma, dbeta

def conv_bn_relu_pool_forward(x, w, b, conv_param, pool_param, gamma, beta, bn_param):
  """Convenience layer that performs a convolution, batchnorm, a ReLU, and a pool.
  Inputs:
  - x: Input to the convolutional layer
  - w, b, conv_param: Weights and parameters for the convolutional layer
  - pool_param: Parameters for the pooling layer
  Returns a tuple of:
  - out: Output from the pooling layer
  - cache: Object to give to the backward pass
  """
  conv, conv_cache = conv_forward_fast(x, w, b, conv_param)
  bn, bn_cache = spatial_batchnorm_forward(conv, gamma, beta, bn_param)
  relu, relu_cache = relu_forward(norm)
  out, pool_cache = max_pool_forward_fast(relu, pool_param)

  cache = (conv_cache, bn_cache, relu_cache, pool_cache)

  return out, cache

def conv_bn_relu_pool_backward(dout, cache):
  """
  Backward pass for the conv-bn-relu-pool convenience layer
  """
  conv_cache, bn_cache, relu_cache, pool_cache = cache

  dpool = max_pool_backward_fast(dout, pool_cache)
  drelu = relu_backward(dpool, relu_cache)
  dbn, dgamma, dbeta = spatial_batchnorm_backward(drelu, bn_cache)
  dx, dw, db = conv_backward_fast(dbn, conv_cache)

  return dx, dw, db, dgamma, dbeta