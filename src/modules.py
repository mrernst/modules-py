# Copyright 2017,2018 Charles Wilmot, Markus Ernst.
# March 2018
# =============================================================================

"""
Simple, yet powerful modular framework for deep learning with tensorflow
See documentation at http://www.github.com/mrernst/modules-py/
"""

# import standard libraries
import tensorflow as tf
import numpy as np

# for cmdline visualization purposes
INDENT = 0


class InputContainer(list):
  """
  InputContainer inherits from list. It allows for appending variables to
  InputContainer with the increment_add operator (+=)
  """
  def __iadd__(self, other):
    self.append(other)
    return self


# Abstract Classes
# -----

class Module:
  """
  Module is an abstract class. It is the base class of all following modules and constitutes
  the very basic properties, IO and naming.

  @param name str, every module must have a name
  """
  def __init__(self, name, *args, **kwargs):
    """
    Creates Module object.

    Args:
      name:                 string, name of the Module
    """
    self.inputs = InputContainer()
    self.outputs = {}
    self.name = name
    # if "name" in kwargs:
    #     self.name = kwargs["name"]
    # else:
    #     raise Exception("module must have a name.")

  def output_exists(self, t):
    """
    output_exists takes a Module object and an integer t and returns true iff
    self.outputs has an entry for timeslice t

    Args:
      t:                    int, indicates the timeslice

    Returns:
      ?:                    bool
    """
    return t in self.outputs

  def need_to_create_output(self, t):
    """
    need_to_create_output takes a Module object and an integer t and returns
    true iff t>=0 and there is not already an output for timeslice t.

    Args:
      t:                    int, indicates the timeslice

    Returns:
      ?:                    bool
    """
    return True if t >= 0 and not self.output_exists(t) else False

  def create_output(self, t):
    """
    create_output takes a Module object and an integer t. It creates outputs for the
    modules using its inputs
    """
    global INDENT
    print("|  " * INDENT + "creating output of {} at time {}".format(self.name, t))
    INDENT += 1
    for inp, dt in self.inputs:
        if inp.need_to_create_output(dt + t):
            inp.create_output(dt + t)
    tensors = self.input_tensors(t)
    self.outputs[t] = self.operation(*tensors)
    INDENT -= 1
    print("|  " * INDENT + "|{}".format(self.outputs[t]))

  def input_tensors(self, t):
    """
    input_tensors takes a Module object and an integer t and aggregates all inputs to
    Module at all timeslices in the future of timeslice t
    """
    return [inp.outputs[t + dt] for inp, dt in self.inputs if t + dt >= 0]


class OperationModule(Module):
  """
  Operation Module is an abstract class. It inherits from Module and can perform
  an operation on the output of another Module in the same time slice.
  To inherit from it, overwrite the 'operation' method in the following way:
  The operation method should take as many parameters as there are modules
  connected to it and return a tensorflow tensor. These parameters are the
  output tensors of the previous modules in the same time slice.
  See the implementation of child classes for more information.
  """

  def add_input(self, other):
    """
    The method add_input connects the module to the output of another module in the
    same time slice.

    Args:
      other:        Module

    Returns:
      self:         OperationModule

    Example usage:
    mw_module.add_input(other_module)
    """
    self.inputs += other, 0
    return self

  def operation(self, x):
    """
    The method operation is supposed to perform an operation on OperationModule's inputs,
    it has to be overwritten when inherting from this abstract class
    """
    raise Exception("Calling abstract class, overwrite this function")



class PlaceholderModule(OperationModule):
  """
  PlaceholderModule is an abstract class. It inherits from OperationModule and takes no input.
  It holds a place where the user can feed in a value to be used in the graph.
  """
  def __init__(self, name, shape, dtype=tf.float32):
    super().__init__(name, shape, dtype)
    self.shape = shape
    self.dtype = dtype
    self.placeholder = tf.placeholder(shape=shape, dtype=dtype, name=self.name)

  def operation(self):
    return self.placeholder



class TimeOperationModule(OperationModule):
  """
  TimeOperationModule is an abstract class. It inherits from OperationModule and can perform
  an operation on the output of another Module in the a different time slice. For Usage, see
  OperationModule. TimeOperationModule and Operationmodule are separated to help the user
  to better keep track of the connectivity of her/his modules.
  """

  def add_input(self, other, t):
    """
    The method add_input connects the module to the output of another module in the
    same OR ANY OTHER time slice. Connecting the module to a future time slice
    (ie t=1, 2...) makes no sense.

    @param other Module, an other module
    @param t int, the delta t which specify to which time slice to connect to

    Example usage:
    same timeslice         mw_module.add_input(other_module,  0)
    previous timeslice     mw_module.add_input(other_module, -1)
    """
    self.inputs += other, t
    return self


class FakeModule(TimeOperationModule):
  """
  FakeModule is an abstract class. It inherits from TimeOperationModule and solely serves
  testing and visualization purposes. Does nothing.
  """
  def operation(self, *args):
    return self.name, args


class VariableModule(OperationModule):
  """
  VariableModule is an abstract class. It inherits from OperationModule and allows storing
  a tensorflow variable in addition to performing an operation. To inherit from it,
  overwrite the 'create_variables' method. The method is then automatically called
  by the constructor.
  See the implementation of child classes for more information.
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.create_variables(self.name + "_var")


  def create_variables(self, name):
    """
    The method create_variables is supposed to create a tensorflow variable,
    it has to be overwritten when inherting from this abstract class

    @param name str, implementation must be changed, name can be accessed from self
    """
    raise Exception("Calling abstract class, overwrite this function")


class ConstantVariableModule(VariableModule):
  """
  ConstantVariableModule inherits from VariableModule. It holds on to a variable that is
  constant in time. Eventual input modules to ConstantVariableModule are disregarded.
  """
  def __init__(self, name, shape, dtype):
    """
    Creates ConstantVariableModule object

    Args:
      name:                 string, name of the module
      shape:                array, shape of the variable to be stored
      dtype:                data type
    """
    self.shape = shape
    self.dtype = dtype
    super().__init__(name, shape, dtype)

  def operation(self, *args):
    """
    operation takes a ConstantVariableModule and returns the tensorflow variable which holds
    which holds the variable created by create_variables
    """
    return self.variable

  def create_variables(self, name):
    """
    create_variables takes a ConstantVariableModule and a name and instatiates a tensorflow variable
    with the shape specified in the constructor.
    It returns nothing

    @param name str, implementation must be changed, name can be accessed from self
    """

    self.variable = tf.Variable(tf.zeros(shape=self.shape, dtype=self.dtype), name=name)



class BiasModule(VariableModule):
  """
  BiasModule inherits from VariableModule. It holds on to a bias variable,
  that can be added to another tensor.
  Eventual input modules to BiasModule are disregarded.
  """

  def __init__(self, name, bias_shape):
    """
    Creates a BiasModule Object

    Args:
      name:                 string, name of the module
      bias_shape:           array, shape of the bias, i.e. (B,W,H,D)
    """
    self.bias_shape = bias_shape
    super().__init__(name, bias_shape)

  def operation(self, *args):
    """
    operation takes a BiasModule and returns the tensorflow variable which holds
    the variable created by create_variables
    """
    return self.bias

  def create_variables(self, name):
    """
    create_variables takes a BiasModule and a name and instantiates a tensorflow variable
    with the shape specified in the constructor.
    It returns nothing.

    @param name str, implementation must be changed, name can be accessed from self
    """
    self.bias = tf.Variable(tf.zeros(shape=self.bias_shape), name=name)


class Conv2DModule(VariableModule):
  """
  Conv2DModule inherits from VariableModule. It takes a single input module
  and performs a convolution.
  """

  def __init__(self, name, filter_shape, strides, padding='SAME'):
    """
    Creates a Conv2DModule object

    Args:
      name:               string, name of the module
      filter_shape:       array, defines the shape of the filter
      strides:            list of ints length 4, stride of the sliding window for each dimension of input
      padding:            string from: "SAME", "VALID", type of padding algorithm to use.

    For more information see tf.nn.conv2d
    """
    self.filter_shape = filter_shape
    super().__init__(name, filter_shape, strides, padding)
    self.strides = strides
    self.padding = padding

  def operation(self, x):
    """
    operation takes a Conv2DModule and x, a 4D tensor and performs a convolution of the input module
    in the current time slice

    Args:
      x:                    4D tensor (BHWD)
    Returns:
      ?
    """
    return tf.nn.conv2d(x, self.weights, strides=self.strides, padding=self.padding, name=self.name)

  def create_variables(self, name):
    """
    create_variables takes a Conv2DModule and a name and instantiates a tensorflow variable for
    the filters (or weights) for the convolution as specified by the parameters given in the constructor.
    """
    self.weights = tf.Variable(tf.truncated_normal(
        shape=self.filter_shape,
        stddev=2 / np.prod(self.filter_shape)), name=name)


class Conv2DTransposeModule(VariableModule):
  """
  Conv2DTransposeModule inherits from VariableModule. It takes a single input module
  and performs a deconvolution.
  """
  def __init__(self, name, filter_shape, strides, output_shape, padding='SAME'):
    """
    Creates a Conv2DTransposeModule object

    Args:
      name:               string, name of the module
      filter_shape:       array, defines the shape of the filter
      output_shape:       array, output shape of the deconvolution op
      strides:            list of ints length 4, stride of the sliding window for each dimension of input
      padding:            string from: "SAME", "VALID", type of padding algorithm to use.

    For more information see tf.nn.conv2d_transpose
    """

    self.filter_shape = filter_shape
    super().__init__(name, filter_shape, strides, output_shape, padding)
    self.strides = strides
    self.output_shape = output_shape
    self.padding = padding


  def operation(self, x):
    """
    operation takes a Conv2DTransposeModule and x, a 4D tensor and performs a deconvolution of the input module
    in the current time slice

    Args:
      x:                    4D tensor (BHWD)
    Returns:
      ?
    """
    return tf.nn.conv2d_transpose(x, self.weights, self.output_shape,
                                    strides=self.strides, padding=self.padding, name=self.name)

  def create_variables(self, name):
    """
    create_variables takes a Conv2DTransposeModule and a name and instantiates a tensorflow variable for
    the filters (or weights) for the deconvolution as specified by the parameters given in the constructor.
    """
    self.weights = tf.Variable(tf.truncated_normal(shape=self.filter_shape, stddev=0.1), name=name)



class MaxPoolingModule(OperationModule):
  """
  MaxPoolingModule inherits from OperationModule. It takes a single input module and
  performs a maxpooling operation
  """
  def __init__(self, name, ksize, strides, padding='SAME'):
    """
    Creates a MaxPoolingModule object

    Args:
      name:                 string, name of the Module
      ksize:
      strides:
      padding:
    """
    super().__init__(name, ksize, strides, padding)
    self.ksize = ksize
    self.strides = strides
    self.padding = padding


  def operation(self, x):
    """
    operation takes a MaxPoolingModule and x, a 4D tensor and performs a maxpooling of
    the input module in the current time slice

    Args:
      x:                    4D tensor (BHWD)
    Returns:
      ?
    """
    return tf.nn.max_pool(x, self.ksize, self.strides, self.padding, name=self.name)



class FlattenModule(OperationModule):
  """
  FlattenModule inherits from OperationModule. It takes a single input module and
  reshapes it. Useful for transfering the output of a convolutional layer to a fully
  connected layer
  """

  def operation(self, x):
    """
    operation takes a FlattenModule and x, a tensor and performs a flattening operation of
    the input module in the current time slice

    Args:
      x:                    tensor
    Returns:
      ret:                  tensor, reshaped tensor x
    """
    ret = tf.reshape(x, (x.shape[0].value, -1), name=self.name)
    return ret



class CropAndConcatModule(TimeOperationModule):
    """
    CropAndConcatModule inherits from TimeOperationModule. It takes exactly 2 input modules,
    crops the output of input module 1 to the size of the output of input module 2 and concatenates
    them along an axis
    """

    def __init__(self, name, axis=3, *args):
      """
      Creates a CropAndConcatModule
      Args:
        name:               string, name of the Module
        axis:               int, dimension along which the concatination takes place
      """
      super().__init__(name)
      self.axis = axis


    def operation(self, x1, x2):
      """
      operation takes a CropAndConcatModule, tensor x1, tensor x2 and crops and concats them together

      Args:
        x1:                 tensor
        x2:                 tensor, same shape as x1
      Returns:
        ?:                  tensor (c,x,y,d), same c,x,y as tensor x1, d tensor x1 + tensor x2
      """
      x1_shape = tf.shape(x1)
      x2_shape = tf.shape(x2)
      # offsets for the top left corner of the crop
      offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
      size = [-1, x2_shape[1], x2_shape[2], -1]
      x1_crop = tf.slice(x1, offsets, size)

      return tf.concat([x1_crop, x2], self.axis)



class FullyConnectedModule(VariableModule):
  """
  FullyConnectedModule inherits from VariableModule. It takes a single module as input
  and performs a basic matrix multiplication
  """
  def __init__(self, name, in_size, out_size):
    """
    Creates FullyConnectedModule object

    Args:
      name:                 string, name of the Module
      in_size:              int, the number of neurons in the previous layer
      out_size:             int, the number of neurons in the current new layer
    """

    self.in_size = in_size
    self.out_size = out_size
    super().__init__(name, in_size, out_size)


  def operation(self, x):
    """
    operation takes a FullyConnectedModule, a tensor x and returns the matrix multiplication
    of the output of the input module by a weight matrix defined by create_variables

    Args:
      x:                  tensor
    Returns:
      ?:                  tensor, matrix multiplication x by a weight matrix
    """
    return tf.matmul(x, self.weights, name=self.name)

  def create_variables(self, name):
    """
    create_variables takes a FullyConnectedModule object and a name and instantiates a tensorflow variable for
    the learnable weights of the matrix as specified by the parameters for sizes given in the constructor.
    """
    self.weights = tf.Variable(tf.truncated_normal(shape=(self.in_size, self.out_size), stddev=0.1), name=name)


class ErrorModule(OperationModule):
  """
  ErrorModule inherits from OperationModule. It takes two modules as input
  and computes an error (or applies any tensorflow operation on two and only
  two tensors)
  """

  def __init__(self, name, error_func):
    """
    Creates ErrorModule object

    Args:
      name:                 string, name of the Module
      error_func:           callable, function that takes exactly 2 args, returns a tf.tensor
    """
    super().__init__(name, error_func)
    self.error_func = error_func

  ## apply the function passed in the constructor its two input modules
  def operation(self, x1, x2):
    """
    operation takes an ErrorModule, a tensor x1, a tensor x2 and returns the output of
    error_func as defined in __init__

    Args:
      x1:                   tensor, logits
      x2:                   tensor, labels
    Returns:
      ?:                    1D tensor, error-value
    """
    return self.error_func(x1, x2, name=self.name)


class DropoutModule(OperationModule):
  """
  DropoutModule inherits from OperationModule. It takes a single module as input
  and applies Dropout to the output of the input module
  """
  def __init__(self, name, keep_prob, noise_shape=None, seed=None):
    """
    Creates DropoutModule object

    Args:
      name:                 string, name of the Module
      keep_prob:            float, the probability that each element is kept.
      noise_shape:          1D int tensor, representing the shape for randomly generated keep/drop flags
      seed:                 int, make errors reproducable by submitting the random seed
    """
    super().__init__(name, keep_prob, noise_shape, seed)
    self.keep_prob = keep_prob
    self.noise_shape = noise_shape
    self.seed = seed

  def operation(self, x):
    """
    operation takes a DropoutModule, a tensor x and returns a tensor the same shape
    with some entries randomly set to zero

    Args:
      x:                    tensor, logits
    Returns:
      ?:                    tensor, same shape as x
    """
    return tf.nn.dropout(x, keep_prob=self.keep_prob, noise_shape=self.noise_shape, seed=self.seed, name=self.name)


class NormalizationModule(OperationModule):
  """
  NormalizationModule inherits from OperationModule. It takes a single module as input
  and applies normalizes it to values between inp_min and inp_max
  """

  def __init__(self, name, inp_max=1, inp_min=-1, dtype=tf.float32):
    """
    Creates NormalizationModule object

    Args:
      name:                 string, name of the Module
      inp_max:              float, maximum of the rescaled range, default 1
      inp_min:              float, minimum of the rescaled range, default -1
      dtype:                type, dtype of the tensor
    """
    super().__init__(name, inp_max, inp_min)
    self.inp_max = inp_max
    self.inp_min = inp_min
    self.dtype = dtype

  def operation(self, x):
    """
    operation takes a NormalizationModule, a tensor x and returns a tensor the same shape
    with values rescaled between inp_max, inp_min

    Args:
      x:                    tensor, RGBA image
    Returns:
      ?:                    tensor, same shape as x
    """
    casted_x = tf.cast(x, dtype=self.dtype)
    rescaled_x = (casted_x / 255) * (self.inp_max - self.inp_min) - self.inp_min
    return rescaled_x


class OptimizerModule(OperationModule):
  """
  OptimizerModule inherits from OperationModule. It takes a single module as input
  and can be used to train a network
  """

  def __init__(self, name, optimizer):
    """
    Creates OptimizerModule object

    Args:
      name:                 string, name of the Module
      optimizer:            tf.train.Optimizer, an instance of an optimizer
    """
    super().__init__(name, optimizer)
    self.optimizer = optimizer

  def operation(self, x):
    """
    operation takes a OptimizerModule, a tensor x and returns the output of the
    input module after adding a dependency in the tensorflow graph.
    Once the output of this module is computed the network is trained

    Args:
      x:                    tensor, most likely the last layer of your network
    Returns:
      ?:                    tensor x
    """
    with tf.control_dependencies([self.optimizer.minimize(x)]):
      ret = tf.identity(x, name=self.name)
    return ret



class BatchAccuracyModule(OperationModule):
  """
  BatchAccuracyModule inherits from OperationModule. It takes a exactly two modules as input
  and computes the classification accuracy
  """

  def operation(self, x1, x2):
    """
    operation takes a BatchAccuracyModule, a tensor x1, a tensor x2 and returns
    computed classification accuracy

    Args:
      x1:                   tensor, prediction of the network
      x2:                   tensor, targets of the supervised task

    Returns:
      ?:                    1D tensor, accuracy
    """
    correct_prediction = tf.equal(tf.argmax(x1, 1), tf.argmax(x2, 1))
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


class NHotBatchAccuracyModule(OperationModule):
  """
  BatchAccuracyModule inherits from OperationModule. It takes a exactly two modules as input
  and computes the classification accuracy for Multi Label Problems
  """

  def __init__(self, name, all_labels_true=True):
    """
    Creates an NHotBatchAccuracyModule object

    Args:
      name:                 string, name of the Module
      all_labels_true:      bool, False: ALL correctly predicted labels are considered for the accuracy
                                  True: Considered only, if all labels of an IMAGE are predicted correctly
    """
    super().__init__(name, all_labels_true)
    self.all_labels_true = all_labels_true


  def operation(self, x1, x2):
    """
    operation takes a BatchAccuracyModule, a tensor x1, a tensor x2 and returns
    a computed classification accuracy

    Args:
      x1:                   tensor, prediction of the network
      x2:                   tensor, multi-hot targets of the supervised task

    Returns:
      accuracy1:            perc. of all labels that are predicted correctly
      accuracy2:            perc. of images where all labels are predicted correctly
    """
    correct_prediction = tf.equal(tf.round(tf.nn.sigmoid(x1)), tf.round(x2))

    if self.all_labels_true:
      all_labels = tf.reduce_min(tf.cast(correct_prediction, tf.float32), 1)
      accuracy2 = tf.reduce_mean(all_labels)
      return accuracy2
    else:
      accuracy1 = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
      return accuracy1

    #return tf.stack([accuracy1, accuracy2]),



class ConstantPlaceholderModule(PlaceholderModule):
  """
  ConstantPlaceholderModule inherits from PlaceholderModule and reserves a place
  to feed in data. Eventual input modules to BiasModule are disregarded.
  """

  def operation(self):
    """
    operation takes a ConstantPlaceholderModule, and always returns the placeholder
    """
    return self.placeholder



class TimeVaryingPlaceholderModule(PlaceholderModule):
  """
  TimeVaryingPlaceholderModule inherits from PlaceholderModule and takes no input.
  reserves a place to feed in data. It remembers the input which
  is fed by the user and rolls it so that at each time slice the network sees
  a new value
  """

  def __init__(self, name, shape, dtype=tf.float32):
    """
    Creates a TimeVaryingPlaceholderModuleme object

    Args:
      name:                 string, name of the Module
      shape:                array, shape of the placeholder
      dtype:                type, dtype of the placeholder
    """
    super().__init__(name, shape, dtype)
    self.outputs[0] = self.placeholder

  def get_max_time(self):
    return len(self.outputs)

  max_time = property(get_max_time)

  # def operation(self, *args):
  #   return self.placeholder

  def need_to_create_output(self, t):
    return True if t >= self.max_time else False

  def shift_by_one(self):
    for i in reversed(range(self.max_time)):
      self.outputs[i + 1] = self.outputs[i]
    self.outputs[0] = self.delayed(self.outputs[1])

  def delayed(self, v):
    v_curr = tf.Variable(tf.zeros(shape=v.shape))
    v_prev = tf.Variable(tf.zeros(shape=v.shape))
    with tf.control_dependencies([v_prev.assign(v_curr)]):
      with tf.control_dependencies([v_curr.assign(v)]):
        v_curr = tf.identity(v_curr)
        v_prev = tf.identity(v_prev)
        return v_prev

  def create_output(self, t):
    global INDENT
    print("|  " * INDENT + "creating output of {} at time {}".format(self.name, t))
    for i in range(t - self.max_time + 1):
      self.shift_by_one()
    print("|  " * INDENT + "|{}".format(self.outputs[t]))



class ActivationModule(OperationModule):
  """
  ActivationModule inherits from OperationModule. It takes a single input module
  and applies and activation function to it
  """
  ## @param activation callable, a tensorflow function
  def __init__(self, name, activation):
    """
    Creates an ActivationModule object

    Args:
      name:                 string, name of the Module
      activation:           callable, tf activation function
    """
    super().__init__(name, activation)
    self.activation = activation

  def operation(self, x):
    """
    operation takes a ActivationModule, a tensor x and returns
    the resulting tensor after applying the activation function

    Args:
      x:                    tensor, preactivation

    Returns:
      ?:                    tensor, same shape as x
    """
    return self.activation(x, name=self.name)



class TimeAddModule(TimeOperationModule):
  """
  TimeAddModule inherits from TimeOperationModule. It can have as many inputs as
  required and sums their outputs. It does support recursions.
  """

  def operation(self, *args):
    """
    operation takes a TimeAddModule and the input modules

    Args:
      *args:                list of modules

    Returns:
      ret:                  tensor, sum of input tensors
    """
    ret = args[0]
    for e in args[1:]:
      ret = tf.add(ret, e, name=self.name)
    return ret



class AddModule(OperationModule):
  """
  AddModule inherits from OperationModule. It can have as many inputs as
  required and sums their outputs. It does not support recursions.
  """
  ## returns the sum of the output tensors of all its input modules
  def operation(self, *args):
    """
    operation takes an AddModule and the input modules

    Args:
      *args:                list of modules

    Returns:
      ret:                    tensor, sum of input tensors
    """
    ret = args[0]
    for e in args[1:]:
      ret = tf.add(ret, e, name=self.name)
    return ret


class BatchNormalizationModule(OperationModule):
  """
  BatchNormalizationModule inherits from OperationModule. It takes a single input module,
  performs Batch normalization and outputs a tensor of the same shape as the input.
  """
  def __init__(self, name, n_out, is_training, beta_init=0.0, gamma_init=1.0, ema_decay_rate=0.5, moment_axes=[0,1,2], variance_epsilon=1e-3):
    """
    Creates a BatchNormalizationModule

    Args:
      name:                tensor, 4D BHWD input
      n_out:               integer, depth of input
      is_training:         boolean tf.Variable, true indicates training phase
      moment_axes:         Array of ints. Axes along which to compute mean and variance.
    """
    super().__init__(name, n_out, is_training, moment_axes, ema_decay_rate)
    self.n_out = n_out
    self.is_training = is_training
    self.moment_axes = moment_axes
    self.ema_decay_rate = ema_decay_rate
    self.variance_epsilon = variance_epsilon

    self.beta = tf.Variable(tf.constant(beta_init, shape=[self.n_out]), name=self.name + '_beta', trainable=True)
    self.gamma = tf.Variable(tf.constant(gamma_init, shape=[self.n_out]), name=self.name + '_gamma', trainable=True)

  def operation(self, x):
    """
    operation takes a BatchNormalizationModule and a 4D BHWD input tensor
    and returns a tensor the same size

    Args:
      x:                   tensor, 4D BHWD input

    Returns:
      ret:                 batch-normalized tensor, 4D BHWD
    """
    #n_out = x.shape[-1]

    # should be only over axis 0, if used for non-conv layers
    batch_mean, batch_var = tf.nn.moments(x, self.moment_axes, name=self.name + '_moments')

    ema = tf.train.ExponentialMovingAverage(decay=self.ema_decay_rate)

    def mean_var_with_update():
      ema_apply_op = ema.apply([batch_mean, batch_var])
      with tf.control_dependencies([ema_apply_op]):
        return tf.identity(batch_mean), tf.identity(batch_var)

    mean, var = tf.cond(self.is_training,
                        mean_var_with_update,
                        lambda: (ema.average(batch_mean), ema.average(batch_var)))

    ret = tf.nn.batch_normalization(x, mean, var, self.beta, self.gamma, self.variance_epsilon)
    return ret



class AbstractComposedModule(Module):
  """
  AbstractComposedModule is an abstract class. It inherits from Module
  and lays the groundwork for a module comprised of other modules
  """
  def __init__(self, *args, **kwargs):
    """
    Creates an AbstractComposedModule Object
    """
    super().__init__(*args, **kwargs)
    self.define_inner_modules(*args, **kwargs)
    self.inputs = self.input_module.inputs
    self.outputs = self.output_module.outputs

  def create_output(self, t):
    """
    create_output takes an AbstractComposesModule object and an integer t.
    It creates outputs for the modules using its inputs
    """
    self.output_module.create_output(t)

  def define_inner_modules(self, *args, **kwargs):
    raise Exception("Calling abstract class, overwrite this function")



class TimeComposedModule(AbstractComposedModule, TimeOperationModule):
  """
  TimeComposedModule is an abstract class. It inherits from AbstractComposedModule and
  TimeOperationModule. It allows when overwritten to create a module that is composed
  of other modules and accept recursions. See the implementation of ConvolutionalLayerModule
  for more info. The method 'define_inner_modules' must be overwritten,
  the attribute input_module must be set to the module which is the input of
  the composed module. The attribute output_module must be set to the module
  which is the output of the composed module
  """
  pass



class ComposedModule(AbstractComposedModule, OperationModule):
  """
   ComposedModule is an abstract class. It inherits from AbstractComposedModule and
   OperationModule. It allows when overwritten to create a module that is
   composed of other modules and does not accept recursions.
   See the implementation of ConvolutionalLayerModule for more info.
   the method 'define_inner_modules' must be overwritten,
   the attribute input_module must be set to the module which is the input of
   the composed module, the attribute output_module must be set to the module
   which is the output of the composed module
  """
  pass



class ConvolutionalLayerModule(ComposedModule):
  """
  ConvolutionalLayerModule inherits from ComposedModule. This composed module
  performs a convolution and applies a bias and an activation function.
  It does not allow recursions
  """
  def define_inner_modules(self, name, activation, filter_shape, strides, bias_shape, padding='SAME'):
    self.input_module = Conv2DModule(name + "_conv", filter_shape, strides, padding=padding)
    self.bias = BiasModule(name + "_bias", bias_shape)
    self.preactivation = AddModule(name + "_preactivation")
    self.output_module = ActivationModule(name + "_output", activation)
    self.preactivation.add_input(self.input_module)
    self.preactivation.add_input(self.bias)
    self.output_module.add_input(self.preactivation)



class ConvolutionalLayerWithBatchNormalizationModule(ComposedModule):
  """
  ConvolutionalLayerWithBatchNormalizationModule inherits from ComposedModule.
  This composed module performs a convolution and applies a bias then Batch
  Normalization and an activation function. It does not allow recursions
  """
  def define_inner_modules(self, name, n_out, is_training, beta_init, gamma_init, ema_decay_rate, activation, filter_shape, strides, bias_shape, padding='SAME'):
    self.input_module = Conv2DModule(name + "_conv", filter_shape, strides, padding=padding)
    #self.bias = BiasModule(name + "_bias", bias_shape)
    #self.preactivation = AddModule(name + "_preactivation")
    self.batchnorm = BatchNormalizationModule(name + "_batchnorm", n_out, is_training, beta_init, gamma_init,
        ema_decay_rate, moment_axes=[0,1,2], variance_epsilon=1e-3)
    self.output_module = ActivationModule(name + "_output", activation)
    #self.preactivation.add_input(self.input_module)
    #self.preactivation.add_input(self.bias)
    self.batchnorm.add_input(self.input_module)
    self.output_module.add_input(self.batchnorm)



class TimeConvolutionalLayerModule(TimeComposedModule):
  """
  TimeConvolutionalLayerModule inherits from TimeComposedModule. This composed module
  performs a convolution and applies a bias and an activation function.
  It does allow recursions
  """
  def define_inner_modules(self, name, activation, filter_shape, strides, bias_shape, padding='SAME'):
    self.input_module = TimeAddModule(name + "_input")
    self.conv = Conv2DModule(name + "_conv", filter_shape, strides, padding=padding)
    self.bias = BiasModule(name + "_bias", bias_shape)
    self.preactivation = AddModule(name + "_preactivation")
    self.output_module = ActivationModule(name + "_output", activation)
    self.conv.add_input(self.input_module)
    self.preactivation.add_input(self.conv)
    self.preactivation.add_input(self.bias)
    self.output_module.add_input(self.preactivation)



class TimeConvolutionalLayerWithBatchNormalizationModule(TimeComposedModule):
  """
  TimeConvolutionalLayerWithBatchNormalizationModule inherits from TimeComposedModule.
  This composed module performs a convolution, applies a bias, batchnormalizes the
  preactivation and then applies an activation function.
  It does allow recursions
  """
  def define_inner_modules(self, name, n_out, is_training, beta_init, gamma_init,
        ema_decay_rate, activation, filter_shape, strides, bias_shape, padding='SAME'):
    self.input_module = TimeAddModule(name + "_input")
    self.conv = Conv2DModule(name + "_conv", filter_shape, strides, padding=padding)
    #self.bias = BiasModule(name + "_bias", bias_shape)
    self.preactivation = AddModule(name + "_preactivation")
    self.batchnorm = BatchNormalizationModule(name + "_batchnorm", n_out, is_training, beta_init, gamma_init,
        ema_decay_rate, moment_axes=[0,1,2], variance_epsilon=1e-3)
    self.output_module = ActivationModule(name + "_output", activation)
    self.conv.add_input(self.input_module)
    self.preactivation.add_input(self.conv)
    #self.preactivation.add_input(self.bias)
    self.batchnorm.add_input(self.preactivation)
    self.output_module.add_input(self.batchnorm)


class FullyConnectedLayerModule(ComposedModule):
  """
  FullyConnectedLayerModule inherits from ComposedModule. This composed module
  performs a full connection and applies a bias and an activation function.
  It does not allow recursions.
  """
  def define_inner_modules(self, name, activation, in_size, out_size):
    self.input_module = FullyConnectedModule(name + "_fc", in_size, out_size)
    self.bias = BiasModule(name + "_bias", (1, out_size))
    self.preactivation = AddModule(name + "_preactivation")
    self.output_module = ActivationModule(name + "_output", activation)
    self.preactivation.add_input(self.input_module)
    self.preactivation.add_input(self.bias)
    self.output_module.add_input(self.preactivation)



class FullyConnectedLayerWithBatchNormalizationModule(ComposedModule):
  """
  FullyConnectedLayerWithBatchNormalizationModule inherits from ComposedModule. This composed module
  performs a full connection and applies a bias batchnormalizes the preactivation and
  applies an activation function. It does not allow recursions.
  """
  def define_inner_modules(self, name, n_out, is_training, beta_init, gamma_init,
        ema_decay_rate, activation, in_size, out_size):
    self.input_module = FullyConnectedModule(name + "_fc", in_size, out_size)
    #self.bias = BiasModule(name + "_bias", (1, out_size))
    self.preactivation = AddModule(name + "_preactivation")
    self.batchnorm = BatchNormalizationModule(name + "_batchnorm", n_out, is_training, beta_init, gamma_init,
        ema_decay_rate, moment_axes=[0], variance_epsilon=1e-3)
    self.output_module = ActivationModule(name + "_output", activation)
    self.preactivation.add_input(self.input_module)
    #self.preactivation.add_input(self.bias)
    self.batchnorm.add_input(self.preactivation)
    self.output_module.add_input(self.batchnorm)

if __name__ == '__main__':

  # visualize recurrent sample structure in cmdline
  f1 = FakeModule("input")
  f2 = FakeModule("conv1")
  f3 = FakeModule("tanh1")
  f4 = FakeModule("conv2")
  f5 = FakeModule("relu2")

  f2.add_input(f1, 0)
  f2.add_input(f2, -1)
  f2.add_input(f4, -1)

  f3.add_input(f2, 0)

  f4.add_input(f3, 0)
  f4.add_input(f4, -1)

  f5.add_input(f4, 0)

  f5.create_output(1)

  fs = [f1, f2, f3, f4, f5]
  for f in fs:
    print(f.name, "---", f.outputs)
