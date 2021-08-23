import matplotlib
matplotlib.use('Agg')
import sys
import os
sys.path.append('../camnet/')
from models import *
import keras
from models import getModel, standardPreprocess
from functions import parseTrainingOptions, parseLoadOptions
import h5py as hd
from tflearn.data_utils import shuffle, to_categorical
import matplotlib.pyplot as plt
import sklearn.metrics
import numpy as np
import sys
from keras.engine import Layer
import keras.backend as K
import numpy as np
import sklearn.metrics
import shutil
import warnings
import tensorflow as tf
from tensorflow.python.ops import array_ops

warnings.filterwarnings("ignore")


class MYEarlyStopping(Callback):
    """Stop training when a monitored metric has stopped improving.
  Assuming the goal of a training is to minimize the loss. With this, the
  metric to be monitored would be `'loss'`, and mode would be `'min'`. A
  `model.fit()` training loop will check at end of every epoch whether
  the loss is no longer decreasing, considering the `min_delta` and
  `patience` if applicable. Once it's found no longer decreasing,
  `model.stop_training` is marked True and the training terminates.
  The quantity to be monitored needs to be available in `logs` dict.
  To make it so, pass the loss or metrics at `model.compile()`.
  Arguments:
    monitor: Quantity to be monitored.
    min_delta: Minimum change in the monitored quantity
        to qualify as an improvement, i.e. an absolute
        change of less than min_delta, will count as no
        improvement.
    patience: Number of epochs with no improvement
        after which training will be stopped.
    verbose: verbosity mode.
    mode: One of `{"auto", "min", "max"}`. In `min` mode,
        training will stop when the quantity
        monitored has stopped decreasing; in `"max"`
        mode it will stop when the quantity
        monitored has stopped increasing; in `"auto"`
        mode, the direction is automatically inferred
        from the name of the monitored quantity.
    baseline: Baseline value for the monitored quantity.
        Training will stop if the model doesn't show improvement over the
        baseline.
    restore_best_weights: Whether to restore model weights from
        the epoch with the best value of the monitored quantity.
        If False, the model weights obtained at the last step of
        training are used.
  Example:
  >>> callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
  >>> # This callback will stop the training when there is no improvement in
  >>> # the validation loss for three consecutive epochs.
  >>> model = tf.keras.models.Sequential([tf.keras.layers.Dense(10)])
  >>> model.compile(tf.keras.optimizers.SGD(), loss='mse')
  >>> history = model.fit(np.arange(100).reshape(5, 20), np.zeros(5),
  ...                     epochs=10, batch_size=1, callbacks=[callback],
  ...                     verbose=0)
  >>> len(history.history['loss'])  # Only 4 epochs are run.
  4
    """

    def __init__(self,
               monitor='val_loss',
               min_delta=0,
               patience=0,
               verbose=0,
               mode='auto',
               baseline=None,
               restore_best_weights=False):
        #super(EarlyStopping, self).__init__()
        #print 'in'
        self.monitor = monitor
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

        if mode not in ['auto', 'min', 'max']:
            logging.warning('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.', mode)
            mode = 'auto'

        if mode == 'min':
              self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

        def on_train_begin(self, logs=None):
            # Allow instances to be re-used
            self.wait = 0
            self.stopped_epoch = 0
            if self.baseline is not None:
                self.best = self.baseline
            else:
                self.best = np.Inf if self.monitor_op == np.less else -np.Inf
            self.best_weights = None

        def on_epoch_end(self, epoch, logs=None):
            current = self.get_monitor_value(logs)
            if current is None:
                return
            if self.monitor_op(current - self.min_delta, self.best):
                self.best = current
                self.wait = 0
                if self.restore_best_weights:
                    self.best_weights = self.model.get_weights()
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
                if self.restore_best_weights:
                    if self.verbose > 0:
                        print('Restoring model weights from the end of the best epoch.')
                    print('Reducing learning rate of 1e-1: {}'.format(initial_lr))
                    self.model.set_weights(self.best_weights)
                    optimizer = self.model.optimizer
                    initial_lr= K.eval(optimizer.lr)
                    initial_lr*=0.5
                    print('Reducing learning rate of 0.5: {}'.format(initial_lr))
                    opt = keras.optimizers.SGD(lr=initial_lr, decay=1e-6, momentum=0.9, nesterov=True)
                    model.compile(optimizer=opt,
                             loss=classifier_loss,
                              metrics=[my_acc_f])
                    #add change learning rate

        def on_train_end(self, logs=None):
            if self.stopped_epoch > 0 and self.verbose > 0:
                print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

        def get_monitor_value(self, logs):
            logs = logs or {}
            monitor_value = logs.get(self.monitor)
            if monitor_value is None:
                logging.warning('Early stopping conditioned on metric `%s` '
                          'which is not available. Available metrics are: %s',
                          self.monitor, ','.join(list(logs.keys())))
            return monitor_value

def unbalanced_classes_loss(y_true, y_pred):
    # compute the number of samples in batch
    batch_size = tf.cast(tf.shape(y_true)[0], tf.float32)
    # function that computes the binary cross entropy loss
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true,logits=y_pred)
    # create a constant = 0 to find all elements within y_true that belong to class 0 (non-tumor)
    zero_class = tf.constant(0, dtype=tf.float32)
    # find elements of class 0 
    where_class_is_zero=tf.cast(tf.equal(y_true, zero_class), dtype=tf.float32)
    # compute the loss weight for class zero
    class_weight_zero = tf.cast(tf.divide(batch_size, 2. * tf.cast(tf.reduce_sum(where_class_is_zero), dtype=tf.float32)+0.001), dtype=tf.float32)
    # find elements of class 1 
    one_class = tf.constant(1, dtype=tf.float32)
    # find elements of class 1
    where_class_is_one=tf.cast(tf.equal(y_true, one_class), dtype=tf.float32)
    # compute the loss weight for class 1
    class_weight_one = tf.cast(tf.divide(batch_size, 2. * tf.cast(tf.reduce_sum(where_class_is_one),dtype=tf.float32)+0.001), dtype=tf.float32)
    #class_weight_vector = [1-ytrue]*[cwzero] + [ytrue][cwone] = A + B
    A = tf.ones(tf.shape(y_true), dtype=tf.float32) - y_true 
    A = tf.scalar_mul(class_weight_zero, A)
    B = tf.scalar_mul(class_weight_one, y_true) 
    class_weight_vector=A+B
    ce = class_weight_vector * ce
    return tf.reduce_mean(ce)



"""
Credits for the Gradient Reversal Layer
https://github.com/michetonu/gradient_reversal_keras_tf/blob/master/flipGradientTF.py
"""
def reverse_gradient(X, hp_lambda):
    '''Flips the sign of the incoming gradient during training.'''
    hp_lambda = hp_lambda
    try:
        reverse_gradient.num_calls += 1
    except AttributeError:
        reverse_gradient.num_calls = 1
    grad_name = "GradientReversal%d" % reverse_gradient.num_calls
    @tf.RegisterGradient(grad_name)
    def _flip_gradients(op, grad):
        grad = tf.negative(grad)
        final_val = grad * hp_lambda 
        return [final_val]
    g = K.get_session().graph
    with g.gradient_override_map({'Identity': grad_name}):
        y = tf.identity(X)
    return y

class GradientReversal(Layer):
    '''Flip the sign of gradient during training.'''
    def __init__(self, hp_lambda, **kwargs):
        super(GradientReversal, self).__init__(**kwargs)
        self.supports_masking = False
        #self.hp_lambda = hp_lambda
        self.hp_lambda = K.variable(hp_lambda, name='hp_lambda')
    def build(self, input_shape):
        self.trainable_weights = []
        return
    def call(self, x, mask=None):
        self.hp_lambda=tf.Print(self.hp_lambda,[self.hp_lambda],'self.hp_lambda: ')
        return reverse_gradient(x, self.hp_lambda)

    def get_output_shape_for(self, input_shape):
        return input_shape
    def get_config(self):
        config = {"name": self.__class__.__name__,
                  'hp_lambda': K.get_value(self.hp_lambda)}
        base_config = super(GradientReversal, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def my_acc_f(y_true, y_pred):
    # we use zero weights to set the loss to zero for unlabeled data
    zero= tf.constant(-1, dtype=tf.float32)
    where = tf.not_equal(y_true, zero)
    where = tf.reshape(where, [-1])
    indices=tf.where(where) #indices where the item of y_true is NOT -1
    indices = tf.reshape(indices, [-1])
    sliced_y_true = tf.nn.embedding_lookup(y_true, indices)
    sliced_y_pred = tf.nn.embedding_lookup(y_pred, indices)
    #sliced_y_true=tf.Print(sliced_y_true, [sliced_y_true], 'sliced_y_true: ')
    #sliced_y_pred=tf.Print(sliced_y_pred, [sliced_y_pred], 'sliced_y_pred: ')
    
    n1 = tf.shape(indices)[0]
    batch_size = tf.shape(y_true)[0]
    #n1=tf.Print(n1, [n1], 'n1: ')
    n2 = batch_size - n1 #number of test images
    sliced_y_true = tf.reshape(sliced_y_true, [n1, -1])
    # the activation function is here.... but this means that I should
    # apply this activation function to the logits also when I run the
    # regression
    
    sliced_y_pred = tf.sigmoid(sliced_y_pred)
    y_pred_rounded = K.round(sliced_y_pred)
    #y_pred_rounded=tf.Print(y_pred_rounded, [y_pred_rounded], 'y_p_rounded: ')
    acc = tf.equal(y_pred_rounded, sliced_y_true)
    #acc = tf.Print(acc, [acc], 'acc: ')
    #acc = tf.reduce_mean(tf.cast(acc, tf.float32))
    #acc = tf.Print(acc, [acc], 'avg acc: ')
    return acc
def my_accuracy(y_true, y_pred):
    sliced_y_pred = tf.sigmoid(y_pred)
    y_pred_rounded = K.round(sliced_y_pred)
    acc = tf.equal(y_pred_rounded, y_true)
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))
    return acc


def tp_count(y_true, y_pred):
    verbose=0
    zero= tf.constant(-1, dtype=tf.float32)
    where = tf.not_equal(y_true, zero)
    where = tf.reshape(where, [-1])
    indices=tf.where(where) #indices where the item of y_true is NOT -1
    indices = tf.reshape(indices, [-1])
    sliced_y_true = tf.nn.embedding_lookup(y_true, indices)
    sliced_y_pred = tf.nn.embedding_lookup(y_pred, indices)
    sliced_y_pred = tf.sigmoid(y_pred)
    y_pred_rounded = K.round(sliced_y_pred)
    
    one_class = tf.constant(1, dtype=tf.float32)
    zero_class = tf.constant(0, dtype=tf.float32)
    tumor_cases_indexes= tf.equal(sliced_y_true, one_class)
    tumor_cases_indexes=tf.where(tumor_cases_indexes)
    #tumor_cases=tf.nn.embedding_lookup(sliced_y_true, tumor_cases_indexes)
    #if verbose:
    #    tumor_cases=tf.Print(tumor_cases,[tumor_cases],'tumor_cases: ')
    predictions_for_those_tumor_cases=tf.nn.embedding_lookup(y_pred_rounded, tumor_cases_indexes)
    if verbose:
        predictions_for_those_tumor_cases=tf.Print(predictions_for_those_tumor_cases,[predictions_for_those_tumor_cases],'predictions_for_those_tumor_cases: ')
    sum_prediction = tf.reduce_sum(predictions_for_those_tumor_cases)
    if verbose:
        sum_prediction=tf.Print(sum_prediction,[sum_prediction],'sum_prediction: ')
        
    # compute the loss weight for class zero
    #true_positive_count=tf.reduce_sum(tf.cast(tf.equal(one_class, predictions_for_those_tumor_cases), dtype=tf.float32))
    
    return sum_prediction

def fp_count(y_true, y_pred):
    zero= tf.constant(-1, dtype=tf.float32)
    where = tf.not_equal(y_true, zero)
    where = tf.reshape(where, [-1])
    indices=tf.where(where) #indices where the item of y_true is NOT -1
    indices = tf.reshape(indices, [-1])
    
    sliced_y_true = tf.nn.embedding_lookup(y_true, indices)
    #sliced_y_true=tf.Print(sliced_y_true,[sliced_y_true], 'sliced_y_true ')
    sliced_y_pred = tf.nn.embedding_lookup(y_pred, indices)
    sliced_y_pred = tf.sigmoid(y_pred)
    y_pred_rounded = K.round(sliced_y_pred)
    #y_pred_rounded=tf.Print(y_pred_rounded,[y_pred_rounded], 'y_pred_rounded ')
    
    one_class = tf.constant(1, dtype=tf.float32)
    zero_class = tf.constant(0, dtype=tf.float32)
    non_tumor_cases_indexes=tf.reshape(tf.equal(sliced_y_true, zero_class), [-1])
    non_tumor_cases_indexes=tf.where(non_tumor_cases_indexes) #indices where the item of y_true is NOT -1
    non_tumor_cases_indexes = tf.reshape(non_tumor_cases_indexes, [-1])
    #tumor_cases_indexes=tf.Print(tumor_cases_indexes,[tumor_cases_indexes], 'tumor_cases_indexes ')
    #tumor_cases=tf.nn.embedding_lookup(sliced_y_true, tumor_cases_indexes)
    
    predictions_for_those_non_tumor_cases=tf.nn.embedding_lookup(y_pred_rounded, non_tumor_cases_indexes)
    #predictions_for_those_tumor_cases=tf.Print(predictions_for_those_tumor_cases,[predictions_for_those_tumor_cases], 'predictions_for_those_tumor_cases ')
    # compute the loss weight for class zero
    false_positive_count=tf.reduce_sum(tf.cast(tf.equal(one_class, predictions_for_those_non_tumor_cases), dtype=tf.float32))
    return false_positive_count

def fn_count(y_true, y_pred):
    zero= tf.constant(-1, dtype=tf.float32)
    where = tf.not_equal(y_true, zero)
    where = tf.reshape(where, [-1])
    indices=tf.where(where) #indices where the item of y_true is NOT -1
    indices = tf.reshape(indices, [-1])
    
    sliced_y_true = tf.nn.embedding_lookup(y_true, indices)
    #sliced_y_true=tf.Print(sliced_y_true,[sliced_y_true], 'sliced_y_true ')
    sliced_y_pred = tf.nn.embedding_lookup(y_pred, indices)
    sliced_y_pred = tf.sigmoid(y_pred)
    y_pred_rounded = K.round(sliced_y_pred)
    #y_pred_rounded=tf.Print(y_pred_rounded,[y_pred_rounded], 'y_pred_rounded ')
    
    one_class = tf.constant(1, dtype=tf.float32)
    zero_class = tf.constant(0, dtype=tf.float32)
    tumor_cases_indexes=tf.reshape(tf.equal(sliced_y_true, one_class), [-1])
    tumor_cases_indexes=tf.where(tumor_cases_indexes) #indices where the item of y_true is NOT -1
    tumor_cases_indexes = tf.reshape(tumor_cases_indexes, [-1])
    #tumor_cases_indexes=tf.Print(tumor_cases_indexes,[tumor_cases_indexes], 'tumor_cases_indexes ')
    #tumor_cases=tf.nn.embedding_lookup(sliced_y_true, tumor_cases_indexes)
    
    predictions_for_those_tumor_cases=tf.nn.embedding_lookup(y_pred_rounded, tumor_cases_indexes)
    #predictions_for_those_tumor_cases=tf.Print(predictions_for_those_tumor_cases,[predictions_for_those_tumor_cases], 'predictions_for_those_tumor_cases ')
    # compute the loss weight for class zero
    true_positive_count=tf.reduce_sum(tf.cast(tf.equal(one_class, predictions_for_those_tumor_cases), dtype=tf.float32))
    false_negative_count=tf.reduce_sum(tf.cast(tf.equal(zero_class, predictions_for_those_tumor_cases), dtype=tf.float32))
    return false_negative_count

def tn_count(y_true, y_pred):
    zero= tf.constant(-1, dtype=tf.float32)
    where = tf.not_equal(y_true, zero)
    where = tf.reshape(where, [-1])
    indices=tf.where(where) #indices where the item of y_true is NOT -1
    indices = tf.reshape(indices, [-1])
    
    sliced_y_true = tf.nn.embedding_lookup(y_true, indices)
    #sliced_y_true=tf.Print(sliced_y_true,[sliced_y_true], 'sliced_y_true ')
    sliced_y_pred = tf.nn.embedding_lookup(y_pred, indices)
    sliced_y_pred = tf.sigmoid(y_pred)
    y_pred_rounded = K.round(sliced_y_pred)
    #y_pred_rounded=tf.Print(y_pred_rounded,[y_pred_rounded], 'y_pred_rounded ')
    
    one_class = tf.constant(1, dtype=tf.float32)
    zero_class = tf.constant(0, dtype=tf.float32)
    non_tumor_cases_indexes=tf.reshape(tf.equal(sliced_y_true, zero_class), [-1])
    non_tumor_cases_indexes=tf.where(non_tumor_cases_indexes) #indices where the item of y_true is NOT -1
    non_tumor_cases_indexes = tf.reshape(non_tumor_cases_indexes, [-1])
    #tumor_cases_indexes=tf.Print(tumor_cases_indexes,[tumor_cases_indexes], 'tumor_cases_indexes ')
    #tumor_cases=tf.nn.embedding_lookup(sliced_y_true, tumor_cases_indexes)
    
    predictions_for_those_non_tumor_cases=tf.nn.embedding_lookup(y_pred_rounded, non_tumor_cases_indexes)
    #predictions_for_those_non_tumor_cases=tf.Print(predictions_for_those_non_tumor_cases,[predictions_for_those_non_tumor_cases], 'predictions_for_those_non_tumor_cases ')
    # compute the loss weight for class zero
    predictions_for_those_non_tumor_cases=tf.zeros(tf.shape(predictions_for_those_non_tumor_cases), dtype=tf.float32)-predictions_for_those_non_tumor_cases
    true_negative_count=tf.reduce_sum(predictions_for_those_non_tumor_cases)
    #true_negative_count=tf.Print(true_negative_count,[true_negative_count], 'true_negative_count ')
    true_negative_count=tf.reduce_sum(tf.cast(tf.equal(zero_class, predictions_for_those_non_tumor_cases), dtype=tf.float32))
    
    return true_negative_count

def classifier_loss(y_true, y_pred):
    # we use zero weights to set the loss to zero for unlabeled data
    verbose=0
    zero= tf.constant(-1, dtype=tf.float32)
    where = tf.not_equal(y_true, zero)
    where = tf.reshape(where, [-1])
    indices=tf.where(where) #indices where the item of y_true is NOT -1
    indices = tf.reshape(indices, [-1])
    sliced_y_true = tf.nn.embedding_lookup(y_true, indices)
    sliced_y_pred = tf.nn.embedding_lookup(y_pred, indices)
    n1 = tf.shape(indices)[0] #number of train images in batch
    batch_size = tf.shape(y_true)[0]
    n2 = batch_size - n1 #number of test images in batch
    sliced_y_true = tf.reshape(sliced_y_true, [n1, -1])
    
    n1_ = tf.cast(n1, tf.float32)
    n2_ = tf.cast(n2, tf.float32)
    multiplier = (n1_+ n2_) / n1_
    zero_class = tf.constant(0, dtype=tf.float32)
    where_class_is_zero=tf.cast(tf.reduce_sum(tf.cast(tf.equal(sliced_y_true, zero_class), dtype=tf.float32)), dtype=tf.float32)
    if verbose:
        where_class_is_zero=tf.Print(where_class_is_zero,[where_class_is_zero],'where_class_is_zero: ')
    class_weight_zero = tf.cast(tf.divide(n1_, 2. * tf.cast(where_class_is_zero, dtype=tf.float32)+0.001), dtype=tf.float32)
    
    if verbose:
        class_weight_zero=tf.Print(class_weight_zero,[class_weight_zero],'class_weight_zero: ')
    one_class = tf.constant(1, dtype=tf.float32)
    where_class_is_one=tf.cast(tf.reduce_sum(tf.cast(tf.equal(sliced_y_true, one_class), dtype=tf.float32)), dtype=tf.float32)
    if verbose:
        where_class_is_one=tf.Print(where_class_is_one,[where_class_is_one],'where_class_is_one: ')
        n1_=tf.Print(n1_,[n1_],'n1_: ')
    class_weight_one = tf.cast(tf.divide(n1_, 2. * tf.cast(where_class_is_one,dtype=tf.float32)+0.001), dtype=tf.float32)
    #
    #
    class_weight_zero =  tf.constant(23477.0/(23477.0+123820.0), dtype=tf.float32)
    class_weight_one =  tf.constant(123820.0/(23477.0+123820.0), dtype=tf.float32)
    #class_weight_one=tf.Print(class_weight_one,[class_weight_one],'class_weight_one: ')
    #class_weight_vector = [1-ytrue]*[cwzero] + [ytrue][cwone] = A + B
    A = tf.ones(tf.shape(sliced_y_true), dtype=tf.float32) - sliced_y_true 
    A = tf.scalar_mul(class_weight_zero, A)
    B = tf.scalar_mul(class_weight_one, sliced_y_true) 
    class_weight_vector=A+B
    #sliced_y_true=tf.Print(sliced_y_true,[sliced_y_true],'sliced_y_true: ')
    #OLD CE
    ce = tf.nn.sigmoid_cross_entropy_with_logits(labels=sliced_y_true,logits=sliced_y_pred)
    #pos_weight=tf.constant(0, dtype=tf.float32)#tf.divide(1)#where_class_is_zero, where_class_is_one+0.001)
    #ce = tf.nn.weighted_cross_entropy_with_logits(sliced_y_true, sliced_y_pred, pos_weight)

    
    #class_weight_vector=tf.Print(class_weight_vector,[class_weight_vector],'class_weight_vector: ')
    ce = tf.multiply(class_weight_vector,ce)
    #ce=tf.Print(ce,[ce],'ce: ')
    #zero_tensor = tf.zeros([n2, 1])
    #final_ce=ce
    #final_ce = tf.cond(n2>0,
    #                   lambda: tf.concat([tf.scalar_mul(multiplier,ce), zero_tensor], axis=0),
    #                   lambda: tf.scalar_mul(multiplier,ce))
    return tf.reduce_mean(ce)

def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.

      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)

      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.

    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)

    """

    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        y_true = tf.cast(y_true, tf.float32)
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        # y_pred = y_pred + epsilon
        # Clip the prediciton value
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss

    return binary_focal_loss_fixed


def tf_focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.

        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.

    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)

    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)

    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)


def r_square(y_true, y_pred):
    from keras import backend as K
    SS_res =  K.sum(K.square(y_true - y_pred))
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )

def get_keys(l):
    db_name=l.split(', ')[0]
    entry_path=l.split(', ')[1].strip()
    patch_no=l.split(', ')[2].strip()
    return db_name, entry_path, patch_no

def get_class(l, entry_path):
    if l.split(', ')[-1].strip('\n')=='test':
        return -1.
    if l.split(', ')[-1].strip('\n')=='test2':
        return -1
    elif l.split(', ')[-1].strip('\n')=='validation':
        return -1
    elif l.split(', ')[-1].strip('\n')=='train':
        if 'normal' in l:
            return 0.
        else:
            return 1.
    else:
        #print('I should not enter here, really: ', l.split(', ')[-1])
        if entry_path.split('/level7')[0]=='normal':
            return 0.
        else:
            return 1.
def get_test_label(entry_path):
    if 'normal' in entry_path:
        return 0.
    else:
        return 1.


def get_domain(db_name, entry_path):
    if db_name=='pannuke':
        return 6
    if db_name=='cam16':
        return 5
    else:
        center_no = entry_path.split('/centre')[1].split('/patient')[0]
        return int(center_no)
    
def zero_loss(y_pred, y_true):
    zero_constant = tf.constant(0, dtype=tf.float32)
    mse = keras.losses.mean_squared_error(y_true, y_pred)
    
    return tf.scalar_mul(zero_constant, mse)
    
def compute_acc(y_pred, y_true):
    #should these be floats?
    y_pred = tf.sigmoid(y_pred)
    y_pred = tf.round(y_pred)
    y_pred = tf.Print(y_pred, [y_pred], 'y_pred: ')
    acc = tf.equal(y_pred, y_true)
    acc = tf.Print(acc, [acc], 'acc equal: ')
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))
    return acc

def auc_from_logits(y_pred, y_true):
    return 1

def compile_model(model, opt, loss=classifier_loss, metrics=[my_acc_f, my_accuracy,tp_count, fp_count, fn_count, tn_count]):
    model.compile(optimizer=opt,
                  loss=loss,
                  metrics=metrics,
                 )
    
class LR_scheduling(Callback):
    def __init__(self,
               new_folder='', loss=classifier_loss, metrics=[my_acc_f, my_accuracy,tp_count, fp_count, fn_count, tn_count]):
        #super(EarlyStopping, self).__init__()
        print 'in'
        #self.monitor = monitor
        self.best_val_loss=10000000000000000000
        self.counter=0
        self.lr_drops_counter=0
        self.new_folder=new_folder
        self.loss=loss
        self.metrics=metrics
        lr_monitor=open('{}/lr_monitor.log'.format(self.new_folder), 'w')
    def on_epoch_end(self, epoch, logs={}):
        optimizer = self.model.optimizer
        initial_lr= K.eval(optimizer.lr)
        lr_decay=K.eval(initial_lr * (1. / (1. + 1e-6 * tf.cast(optimizer.iterations, tf.float32))))
        lr_monitor=open('{}/lr_monitor.log'.format(self.new_folder), 'a')
        lr_monitor.write('epoch: {}, lr: {}, decayed: {} \n'.format(epoch, initial_lr, lr_decay))
        print 'LR: {},  decay: {}'.format(initial_lr, lr_decay)
        lr_monitor.close()
        print 'val_loss: {}'.format(logs['val_loss'])
        """
        if logs['val_loss']<self.best_val_loss:
            self.best_val_loss=logs['val_loss']
        else: 
            self.counter+=1
        if self.counter==2:
            #print 'reducing_lr'
            self.counter=0
            lr=initial_lr*0.5
            self.lr_drops_counter+=1
            opt = keras.optimizers.SGD(lr=lr, momentum=0.9, nesterov=False)
            compile_model(self.model, opt, loss=self.loss, metrics=self.metrics)
            lr_decay=K.eval(lr * (1. / (1. + 1e-6 * tf.cast(optimizer.iterations, tf.float32))))
            print 'Reducing LR to: {},  decayed: {}'.format(lr, lr_decay)
        if self.lr_drops_counter==5:
            self.lr_drops_counter=0
            lr=initial_lr*10
            opt = keras.optimizers.SGD(lr=lr, momentum=0.9, nesterov=False)
            compile_model(self.model, opt, loss=self.loss, metrics=self.metrics)
            lr_decay=K.eval(lr * (1. / (1. + 1e-6 * tf.cast(optimizer.iterations, tf.float32))))
            print 'Restarting LR to: {},  decayed: {}'.format(lr, lr_decay)"""