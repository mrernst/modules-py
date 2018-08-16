class PreprocessModule(ComposedModule):
  """
  ConvolutionalLayerModule inherits from ComposedModule. This composed module
  performs a convolution and applies a bias and an activation function.
  It does not allow recursions
  """
  def define_inner_modules(self, name, image_width, angle_max, brightness_max_delta, contrast_lower, contrast_upper, hue_max_delta):
    self.input_module = RotateImageModule('', angle_max)
    self.centercrop = CropModule('', image_width//10*5, image_width//10*5)
    self.imagestats = RandomImageStatsModule('', brightness_max_delta, contrast_lower, contrast_upper, hue_max_delta)
    self.output_module = RandomCropModule('', image_width//10*4, image_width//10*4)
    # wire modules
    self.centercrop.add_input(self.input_module)
    self.imagestats.add_input(self.centercrop)
    self.output_module.add_input(self.imagestats)


class RotateImageModule(OperationModule):
  """
  RotateImageModule inherits from OperationModule. It takes a single module as input
  and …
  """

  def __init__(self, name, angle_max, random_seed = None):
    """
    Creates RotateImageModule object

    Args:
      name:                        string, name of the Module
      angle_max:                   float, angle at 1 sigma
      random_seed:                 int, An operation-specific seed
    """
    super().__init__(name, angle_max, random_seed)
    self.angle_max = angle_max
    self.random_seed = random_seed

  def operation(self, x):
    """
    operation takes a RotateImageModule, a tensor x and returns a tensor the same shape
    …

    Args:
      x:                    tensor, RGBA image
    Returns:
      ?:                    tensor, same shape as x
    """
    batch_size = x[0]
    
    angles = self.angle_max * tf.random_normal(
        batch_size,
        mean=0.0,
        stddev=1.0,
        dtype=tf.float32,
        seed=self.random_seed,
        name=None
    )
    
    rotated_x = tf.contrib.image.rotate(
        x,
        angles,
        interpolation='NEAREST',
        name=None
    )
    
    return rotated_x


class RandomImageStatsModule(OperationModule):
  """
  RandomImageStatsModule inherits from OperationModule. It takes a single module as input
  and …
  """

  def __init__(self, name, brightness_max_delta, contrast_lower, contrast_upper, hue_max_delta, random_seed = None):
    """
    Creates RandomImageStatsModule object

    Args:
      name:                        string, name of the Module
      brightness_max_delta:        float, must be non-negative.
      contrast_lower:              float, Lower bound for the random contrast factor.
      contrast_upper:              float, Upper bound for the random contrast factor.
      hue_max_delta:               float, Maximum value for the random delta.
      random_seed:                 int, An operation-specific seed
    """
    super().__init__(name, brightness_max_delta, contrast_lower, contrast_upper,hue_max_delta, random_seed)
    self.brightness_max_delta = brightness_max_delta
    self.contrast_lower = contrast_lower
    self.contrast_upper = contrast_upper
    self.hue_max_delta = hue_max_delta
    self.random_seed = random_seed

  def operation(self, x):
    """
    operation takes a RandomImageStatsModule, a tensor x and returns a tensor the same shape
    …

    Args:
      x:                    tensor, RGBA image
    Returns:
      ?:                    tensor, same shape as x
    """
    
    preprocessed_x = tf.image.random_brightness(
        x,
        self.brightness_max_delta,
        seed=self.random_seed
    )
    
    preprocessed_x = tf.image.random_contrast(
        preprocessed_x,
        self.contrast_lower,
        self.contrast_upper,
        seed=self.random_seed
    )
    
    preprocessed_x = tf.image.random_hue(
        preprocessed_x,
        self.hue_max_delta,
        seed=self.random_seed
    )
    
    preprocessed_x = tf.image.random_flip_left_right(
        preprocessed_x,
        seed=self.random_seed
    )
    
    return preprocessed_x


class RandomCropModule(OperationModule):
  """
  RandomCropModule inherits from OperationModule. It takes a single input module and
  resizes it.
  """
  def __init__(self, name, height, width):
     """
     Creates a RandomCropModule object

     Args:
       name:                 string, name of the Module
       height:               int, desired output image height
       weight:               int, desired output image width
     """
     super().__init__(name, height, width)
     self.height = height
     self.width = width

  def operation(self, x):
    """
    operation takes a RandomCropModule and x, a tensor and performs a cropping operation of
    the input module in the current time slice

    Args:
      x:                    tensor
    Returns:
      ret:                  tensor, (B,self.height,self.width,D)
    """
    batchsize = x[0]
    channels = x[-1]
    ret = tf.random_crop(x, [batchsize, self.height, self.width, channels])
    return ret
