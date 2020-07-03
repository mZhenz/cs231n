import numpy as np
from random import shuffle

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in range(num_train):  #训练样本个数
    scores = X[i].dot(W)  #1*3072 mul 3072*10 = 1*10
    correct_class_score = scores[y[i]]  #获得Syi
    ds_w = np.repeat(X[i], num_classes).reshape(-1, num_classes)  #计算偏S偏w, 3072*10
    dm_s = np.zeros(W.shape)  #初始化dm/ds
    for j in range(num_classes):
      if j == y[i]:
        continue
      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:  #max(0, sj-syi+1)
        dm_s[:,j] = 1   #这两步计算偏margin偏s
        dm_s[:,y[i]] -= 1
        loss += margin
    dW_i = ds_w * dm_s  #计算偏margin偏w，点乘，单个输入
    dW += dW_i  #求得总的梯度

  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train

  # Add regularization to the loss.正则化项
  loss += 0.5 * reg * np.sum(W * W) #L2正则化
  dW += W*2

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  scores = X.dot(W)
  correct_class_scores = scores[np.arange(num_train), y] #从二维矩阵中抽取出真分数点
  correct_class_scores = np.reshape(correct_class_scores, (num_train, -1))
  margin = scores - correct_class_scores + 1
  margin[np.arange(num_train), y] = 0
  margin[margin<=0] = 0.0
  loss += np.sum(margin)/num_train
  loss += 0.5 * reg * np.sum(W * W)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  margin[margin>0] = 1.0
  row_sum = np.sum(margin, axis=1)
  margin[np.arange(num_train), y] = -row_sum
  dW = 1.0/num_train * np.dot(X.T, margin) + reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
