import tensorflow as tf
import numpy as np


INDENT = 0


class InputContainer(list):
    def __iadd__(self, other):
        self.append(other)
        return self


## Base class of a module.
#  @param name str, every module must have a name
class Module:
    def __init__(self, name, *args, **kwargs):
        self.inputs = InputContainer()
        self.outputs = {}
        self.name = name
        # if "name" in kwargs:
        #     self.name = kwargs["name"]
        # else:
        #     raise Exception("modules must have a name.")

    def output_exists(self, t):
        return t in self.outputs

    def need_to_create_output(self, t):
        return True if t >= 0 and not self.output_exists(t) else False

    def create_output(self, t):
        global INDENT
        print("|   " * INDENT + "creating output of {} at time {}".format(self.name, t))
        INDENT += 1
        for inp, dt in self.inputs:
            if inp.need_to_create_output(dt + t):
                inp.create_output(dt + t)
        tensors = self.input_tensors(t)
        self.outputs[t] = self.operation(*tensors)
        INDENT -= 1
        print("|   " * INDENT + "|{}".format(self.outputs[t]))

    def input_tensors(self, t):
        return [inp.outputs[t + dt] for inp, dt in self.inputs if t + dt >= 0]


## A placeholder module is a module that takes no input. It holds a place where
#  the user can feed in a value to be computed in the graph
class PlaceholderModule(Module):
    def __init__(self, name, shape, dtype=tf.float32):
        super().__init__(name, shape, dtype)
        self.shape = shape
        self.dtype = dtype
        self.placeholder = tf.placeholder(shape=shape, dtype=dtype, name=self.name)


## An operation module is a module that can perform an operation on the output
#  of an other module in the same time slice. This class is an abstract class.
#  To inherit from it, overwrite the 'operation' method in the following way:
#  the operation method should take as many parameters as there are modules
#  connected to it and return a tensorflow tensor. These parameters are the
#  output tensors of the previous modules in the same time slice.
#  See the implementation of child classes for more information
class OperationModule(Module):
    ## Method to connect the module to the output of an other module in the
    #  same time slice.
    #  mw_module.add_input(other_module)
    def add_input(self, other):
        self.inputs += other, 0
        return self

    ## This method has to be overwritten
    def operation(self, x):
        raise Exception("Calling abstract class, overwrite this function")


## Same as an OperationModule but this class allows to connect itself with the
#  output of an other module in a different time slice. Those two kinds of
#  modules are separated to help the developer to keep the connectivity of his
#  modules clear.
class TimeOperationModule(OperationModule):
    ## method to connect the module to the output of an other module in the
    #  same OR ANY OTHER time slice.
    #  mw_module.add_input(other_module,  0) # same time slice
    #  mw_module.add_input(other_module, -1) # previous time slice
    #  connecting the module to a future time slice (ie t=1, 2...) makes no sens
    #  @param other Module, an other module
    #  @param t int, the delta t which specify to which time slice to connect to
    def add_input(self, other, t):
        self.inputs += other, t
        return self

## For testing purpose only, does nothing
class FakeModule(TimeOperationModule):
    def operation(self, *args):
        return self.name, args

## This class allows a module to hold a tensorflow variable. To do so, you have
#  to overwrite the 'create_variables' method. It is then automatically called
#  by the constructor. See child classes for more details.
class VariableModule(OperationModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.create_variables(self.name + "_var")

    ## method to create tensorflow variables. This method must be overwritten
    #  @param name str, implementation must be changed, the name can be accessed
    #  from self
    def create_variables(self, name):
        raise Exception("Calling abstract class, overwrite this function")


## Module to create a bias, that can be added to an other tensor
#  inputs modules of this module are not taken into account
class BiasModule(VariableModule):
    ## @param bias_shape tuple, list or np.array, the shape of the bias
    def __init__(self, name, bias_shape):
        self.bias_shape = bias_shape
        super().__init__(name, bias_shape)

    ## Always return the tensorflow Variable which holds the bias
    def operation(self, *args):
        return self.bias

    ## instanciate the tensorflow Varaible with the shape specified in the
    #  constructor
    def create_variables(self, name):
        self.bias = tf.Variable(tf.zeros(shape=self.bias_shape), name=name)


## Module to perform a convolution. This module can have a single input module
class Conv2DModule(VariableModule):
    ## It takes the same argument as the tensorflow conv2d
    def __init__(self, name, filter_shape, strides, padding='SAME'):
        self.filter_shape = filter_shape
        super().__init__(name, filter_shape, strides, padding)
        self.strides = strides
        self.padding = padding

    ## Perform a convolution on the output tensor of the input module in the
    #  current time slice.
    #  This module can have a single input module.
    def operation(self, x):
        return tf.nn.conv2d(x, self.weights, strides=self.strides, padding=self.padding, name=self.name)

    ## Creates the filters (or weights) for the convolution according to the
    #  parameters given in the constructor
    def create_variables(self, name):
        self.weights = tf.Variable(tf.truncated_normal(
            shape=self.filter_shape,
            stddev=2 / np.prod(self.filter_shape)), name=name)


## Module to perform a deconvolution. This module can have a single input module
class Conv2DTransposeModule(VariableModule):
    ## It takes the same argument as the tensorflow conv2d_transpose
    def __init__(self, name, filter_shape, strides, output_shape, padding='SAME'):
        self.filter_shape = filter_shape
        super().__init__(name, filter_shape, strides, output_shape, padding)
        self.strides = strides
        self.output_shape = output_shape
        self.padding = padding

    ## Perform a deconvolution on the output tensor of the input module in the
    #  current time slice.
    #  This module can have a single input module.
    def operation(self, x):
        return tf.nn.conv2d_transpose(x, self.weights, self.output_shape,
                                      strides=self.strides, padding=self.padding, name=self.name)

    ## Creates the filters (or weights) for the deconvolution according to the
    #  parameters given in the constructor
    def create_variables(self, name):
        self.weights = tf.Variable(tf.truncated_normal(shape=self.filter_shape, stddev=0.1), name=name)


## Module to perform a maxpooling. This module can have a single input module
class MaxPoolingModule(OperationModule):
    def __init__(self, name, ksize, strides, padding='SAME'):
        super().__init__(name, ksize, strides, padding)
        self.ksize = ksize
        self.strides = strides
        self.padding = padding

    ## Perform a maxpooling on the output tensor of the input module in the
    #  current time slice.
    #  This module can have a single input module.
    def operation(self, x):
        return tf.nn.max_pool(x, self.ksize, self.strides, self.padding, name=self.name)


## Module to flatten the output of the input module. It useful for example to
#  perform a fully connected layer after a convolutional layer
class FlattenModule(OperationModule):
    ## Returns the reshaped output tensor of the input module
    def operation(self, x):
        return tf.reshape(x, (x.shape[0].value, -1), name=self.name)


## Returns element wise multiplication of inputs
class EleMultiModule(TimeOperationModule):
    def operation(self, *args):
        x = args[0]
        for y in args[1:]:
            x = tf.multiply(x, y, name=self.name)
        return x

class ConcatModule(TimeOperationModule):
    def __init__(self, name, axis, resulting_axis_shape):
        self.axis = axis
        self.resulting_axis_shape = resulting_axis_shape
        super().__init__(name, axis)


    # Return concatenated vector
    def operation(self, *args):
        new_axis_shape = sum([t.shape[self.axis].value for t in args])
        if new_axis_shape > self.resulting_axis_shape:
            raise Exception("Inputs are too big!")
        if new_axis_shape < self.resulting_axis_shape:
            shape = [s.value for s in args[0].shape]
            shape[self.axis] = self.resulting_axis_shape - new_axis_shape
            args = list(args)
            args.append(tf.zeros(shape=shape, dtype=args[0].dtype))
        return tf.concat(list(args), self.axis, name=self.name)  # stack them vertically



## Module to perform a matrix multiplication. Again, this module takes a single
#  module as input.
class FullyConnectedModule(VariableModule):
    ## @param in_size int, the number of neurones in the previous layer
    ## @param out_size int, the number of neurones in the current new layer
    def __init__(self, name, in_size, out_size):
        self.in_size = in_size
        self.out_size = out_size
        super().__init__(name, in_size, out_size)

    ## Returns the matrix multiplication of the output of the input module by a
    #  weight matrix.
    def operation(self, x):
        return tf.matmul(x, self.weights, name=self.name)

    ## Instanciate a weight matrix accordingly to the sizes specified in the
    #  constructor
    def create_variables(self, name):
        self.weights = tf.Variable(tf.truncated_normal(shape=(self.in_size, self.out_size), stddev=0.1), name=name)


## Module to compute an error (or apply any tensorflow operation on two and only
#  two tensors)
class ErrorModule(OperationModule):
    ## @param error_func callable, function that takes exactly 2 arguments and
    #  returns a tensorflow tensor.
    def __init__(self, name, error_func):
        super().__init__(name, error_func)
        self.error_func = error_func

    ## apply the function passed in the constructor its two input modules
    def operation(self, x1, x2):
        return self.error_func(x1, x2, name=self.name)

## Module to add Dropout to a convolutional layer. This module takes a single
# input module
class DropoutModule(OperationModule):
    def __init__(self, name, keep_prob, noise_shape=None, seed=None):
        super().__init__(name, keep_prob, noise_shape, seed)
        self.keep_prob = keep_prob
        self.noise_shape = noise_shape
        self.seed = seed

    def operation(self, x):
        return tf.nn.dropout(x, keep_prob=self.keep_prob, noise_shape=self.noise_shape,
            seed=self.seed, name=self.name)

## Module to train a network. This module takes a single input module.
class OptimizerModule(OperationModule):
    ## @param optimizer tf.train.Optimizer, an instance of an optimizer
    def __init__(self, name, optimizer):
        super().__init__(name, optimizer)
        self.optimizer = optimizer

    ## returns the output of the input module after adding a dependency in the
    #  tensorflow graph. Once the output of this module is computed the network
    #  is trained
    def operation(self, x):
        with tf.control_dependencies([self.optimizer.minimize(x)]):
            ret = tf.identity(x, name=self.name)
        return ret


## Module to compute the classification accuracy. This modules takes exactly
#  two input modules (two one-hot tensors to compare)
class BatchAccuracyModule(OperationModule):
    ## returns a tensorflow tensor which holds the accuracy (scalar)
    def operation(self, x1, x2):
        correct_prediction = tf.equal(tf.argmax(x1, 1), tf.argmax(x2, 1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


## Module to reserve a place to feed in data. This module takes no input
class ConstantPlaceholderModule(PlaceholderModule):
    ## always returns the placeholder
    def operation(self):
        return self.placeholder


## Module to reserve a place to feed in data. It remembers the input which
# is fed by the user and roll it so that at each time slice the network sees
# a new value. This module takes no input.
class TimeVaryingPlaceholderModule(PlaceholderModule):
    def __init__(self, name, shape, dtype=tf.float32):
        super().__init__(name, shape, dtype)
        self.outputs[0] = self.placeholder

    def get_max_time(self):
        return len(self.outputs)

    max_time = property(get_max_time)

    # def operation(self, *args):
    #     return self.placeholder

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
        print("|   " * INDENT + "creating output of {} at time {}".format(self.name, t))
        for i in range(t - self.max_time + 1):
            self.shift_by_one()
        print("|   " * INDENT + "|{}".format(self.outputs[t]))


## Apply an activation function on its input module
class ActivationModule(OperationModule):
    ## @param activation callable, a tensorflow function
    def __init__(self, name, activation):
        super().__init__(name, activation)
        self.activation = activation

    ## returns the resulting tensor after applying the activation function
    def operation(self, x):
        return self.activation(x, name=self.name)


## This module computes the sum of all its input. It can have as many inputs
#  as requiered and support recursions
class TimeAddModule(TimeOperationModule):
    ## returns the sum of the output tensors of all its input modules
    def operation(self, *args):
        return sum(args)


## This module computes the sum of all its input. It can have as many inputs
#  as requiered and does not support recursions
class AddModule(OperationModule):
    ## returns the sum of the output tensors of all its input modules
    def operation(self, *args):
        return sum(args)


class AbstractComposedModule(Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.define_inner_modules(*args, **kwargs)
        self.inputs = self.input_module.inputs
        self.outputs = self.output_module.outputs

    def create_output(self, t):
        self.output_module.create_output(t)

    def define_inner_modules(self, *args, **kwargs):
        raise Exception("Calling abstract class, overwrite this function")


## This class is an abstract class which allows when overwritten to create a
#  module that is composed of other modules and accept recursions
#  See the implementation of ConvolutionalLayerModule for more info.
#  the method 'define_inner_modules' must be overwritten,
#  the attribute input_module must be set to the module which is the input of
#  the composed module
#  the attribute output_module must be set to the module which is the output of
#  the composed module
class TimeComposedModule(AbstractComposedModule, TimeOperationModule):
    pass


## This class is an abstract class which allows when overwritten to create a
#  module that is composed of other modules and does not accept recursions
#  See the implementation of ConvolutionalLayerModule for more info.
#  the method 'define_inner_modules' must be overwritten,
#  the attribute input_module must be set to the module which is the input of
#  the composed module
#  the attribute output_module must be set to the module which is the output of
#  the composed module
class ComposedModule(AbstractComposedModule, OperationModule):
    pass


## This composed module performs a convolution and applies a bias and an
#  activation function. It does not allow recursions
class ConvolutionalLayerModule(ComposedModule):
    def define_inner_modules(self, name, activation, filter_shape, strides, bias_shape, padding='SAME'):
        self.input_module = Conv2DModule(name + "_conv", filter_shape, strides, padding=padding)
        self.bias = BiasModule(name + "_bias", bias_shape)
        self.preactivation = AddModule(name + "_preactivation")
        self.output_module = ActivationModule(name + "_output", activation)
        self.preactivation.add_input(self.input_module)
        self.preactivation.add_input(self.bias)
        self.output_module.add_input(self.preactivation)


## This composed module performs a convolution and applies a bias and an
#  activation function. It does allow recursions
class TimeConvolutionalLayerModule(TimeComposedModule):
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


## This composed module performs a full connection and applies a bias and an
#  activation function. It does not allow recursions
class FullyConnectedLayerModule(ComposedModule):
    def define_inner_modules(self, name, activation, in_size, out_size):
        self.input_module = FullyConnectedModule(name + "_fc", in_size, out_size)
        self.bias = BiasModule(name + "_bias", (1, out_size))
        self.preactivation = AddModule(name + "_preactivation")
        self.output_module = ActivationModule(name + "_output", activation)
        self.preactivation.add_input(self.input_module)
        self.preactivation.add_input(self.bias)
        self.output_module.add_input(self.preactivation)


if __name__ == '__main__':
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

    # inp = InputModule(shape=(10, 8, 8, 1), name="input")
    # conv1 = Conv2DModule((4,4,1,3), (1,1,1,1), name="conv1")
    # conv2 = Conv2DModule((4,4,3,3), (1,1,1,1), name="conv2")
    # bias1 = BiasModule((3,), name="bias1")
    # add1 = AddModule(name="add1")
    # tanh1 = ActivationModule(tf.tanh, name="tanh1")
    #
    #
    # conv1 += inp, 0
    # add1 += conv1, 0
    # add1 += conv2, 0
    # add1 += bias1, 0
    # conv2 += add1, -1
    # tanh1 += add1, 0
    # tanh1.create_output(3)