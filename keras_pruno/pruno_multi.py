import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.framework import smart_cond

import numpy as np


def pruno_multi_channels_first(similarity, inputs_flat, actual_batchsize, fmap_count, fmap_size):
    flatshape = (-1, fmap_count, fmap_size)
    fmap_mean = tf.math.reduce_mean(inputs_flat, axis=-1, keepdims=True)
    live = tf.cast(inputs_flat[:, :, :] > fmap_mean[:, :, :], dtype='float32')
    mult_list = []
    for fm_x in range(fmap_count):
        for fm_y in range(fm_x+1, fmap_count):
            mult_list.append(live[:, fm_x, :] * live[:, fm_y, :])
    mult = tf.stack(mult_list, axis=1)
    percent = tf.math.reduce_sum(mult, axis=-1, keepdims=True) / flatshape[2]
    mask = tf.cast(percent < similarity, dtype='float32')
    mask_unstack = tf.unstack(mask, axis=1)
    reduce_list = [None] * fmap_count
    for i in range(fmap_count):
        reduce_list[i] = []
    counter = 0
    for fm_x in range(fmap_count):
        for fm_y in range(fm_x+1, fmap_count):
            mask_cell = mask_unstack[counter]
            reduce_list[fm_x].append(mask_cell)
            reduce_list[fm_y].append(mask_cell)
            counter += 1
    for i in range(fmap_count):
        for r in range(len(reduce_list[i]), fmap_count):
            one = tf.ones_like(reduce_list[0][0])
            reduce_list[i].append(one)
    for i in range(fmap_count):
        row = tf.stack(reduce_list[i], axis=1)
        reduce_list[i] = tf.math.reduce_min(row, axis=1)
    reduce_mask = tf.stack(reduce_list, axis=1)
    return inputs_flat * reduce_mask

def pruno_multi_channels_last(similarity, inputs_flat, actual_batchsize, fmap_count, fmap_size):
    print('pruno_multi_channels_last:', similarity, inputs_flat, actual_batchsize, fmap_count, fmap_size)
    flatshape = (-1, fmap_size, fmap_count)
    fmap_mean = tf.math.reduce_mean(inputs_flat, axis=1, keepdims=True)
    live = tf.cast(inputs_flat[:, :, :] > fmap_mean[:, :, :], dtype='float32')
    mult_list = []
    print('live:', live)
    for fm_x in range(fmap_count):
        for fm_y in range(fm_x+1, fmap_count):
            mult_list.append(live[:, :, fm_x] * live[:, :, fm_y])
    print('mult_list:', mult_list)
    mult = tf.stack(mult_list, axis=2)
    print('mult:', mult)
    percent = tf.math.reduce_sum(mult, axis=1, keepdims=True) / flatshape[1]
    print('percent:', percent)
    mask = tf.cast(percent < similarity, dtype='float32')
    print('mask:', mask)
    # return mask
    mask_unstack = tf.unstack(mask, axis=2)
    print('mask_unstack:', mask_unstack)
    reduce_list = [None] * fmap_count
    for i in range(fmap_count):
        reduce_list[i] = []
    counter = 0
    for fm_x in range(fmap_count):
        for fm_y in range(fm_x+1, fmap_count):
            mask_cell = mask_unstack[counter]
            reduce_list[fm_x].append(mask_cell)
            reduce_list[fm_y].append(mask_cell)
            counter += 1
    print('Final: x, y, counter:', fm_x, fm_y, counter)
    for i in range(fmap_count):
        for r in range(len(reduce_list[i]), fmap_count):
            one = tf.ones_like(reduce_list[0][0])
            reduce_list[i].append(one)
    for i in range(fmap_count):
        row = tf.stack(reduce_list[i], axis=1)
        reduce_list[i] = tf.math.reduce_min(row, axis=1)
    print('reduce_list[0]:', reduce_list[0])
    reduce_mask = tf.stack(reduce_list, axis=1)
    print('inputs_flat:', inputs_flat)
    reduce_mask = tf.reshape(reduce_mask, (-1, 1, fmap_count))
    return reduce_mask
    print('reduce_mask:', reduce_mask)
    return inputs_flat * reduce_mask


class Pruno2DMulti(tf.keras.layers.Layer):
  """Applies Pruning Dropout to the input.
  The Pruno2D layer compares all possible pairs of feature maps, and sets
  both feature maps to zero when they are "too similar". Similarity is measured
  by counting the pixels in both feature maps that are greater than the mean 
  of each feature map.
  When using `model.fit`,
  `training` will be appropriately set to True automatically, and in other
  contexts, you can set the kwarg explicitly to True when calling the layer.
  (This is in contrast to setting `trainable=False` for a Pruno layer.
  `trainable` does not affect the layer's behavior, as Pruno does
  not have any variables/weights that can be frozen during training.)
  >>> layer = Pruno(.2, seed=0, input_shape=(2, 5, 2))
  >>> data = np.arange(20).reshape(2, 5, 2).astype(np.float32)
  >>> print(data)
  [[[ 0.  1.]
  [ 2.  3.]
  [ 4.  5.]
  [ 6.  7.]
  [ 8.  9.]]

 [[10. 11.]
  [12. 13.]
  [14. 15.]
  [16. 17.]
  [18. 19.]]]
  >>> outputs = layer(data, training=True)
  >>> print(outputs)
  tf.Tensor(
  [[ 0.    1.25]
   [ 2.5   3.75]
   [ 5.    6.25]
   [ 7.5   8.75]
   [10.    0.  ]], shape=(5, 2), dtype=float32)
  Arguments:
    rate: Float between 0 and 1. 1.0 - rate = percentage of matching values 
    which triggers a dropout event
    seed: A Python integer to use as random seed.
  Call arguments:
    inputs: Input tensor (of any rank).
    training: Python boolean indicating whether the layer should behave in
      training mode (adding dropout) or in inference mode (doing nothing).
  """

  def __init__(self, similarity, noise_shape=None, seed=None, **kwargs):
    super(Pruno2DMulti, self).__init__(**kwargs)
    if similarity < 0.0 or similarity > 1.0:
        raise ValueError('similarity must be between 0.0 and 1.0: %s' % str(similarity))
    self.similarity = similarity
    self.supports_masking = True

  def build(self, input_shape):
    shape = input_shape.as_list()
    self.fmap_count = shape[1]
    self.fmap_even = (shape[1]//2)*2
    self.fmap_shape = (shape[2], shape[3])
    self.built = True

  def call(self, inputs, training=None):
    if training is None:
      training = K.learning_phase()
    
    training = True

    def identity_inputs():
        return inputs

    def dropped_inputs():
      actual_batchsize = tf.shape(inputs)[0:1]
      input_shape = (-1, self.fmap_count, self.fmap_shape[0], self.fmap_shape[1])
      flatshape = (-1, self.fmap_count, self.fmap_shape[0] * self.fmap_shape[1])
      inputs_flatmap = tf.reshape(inputs, flatshape)
      outputs_flat = pruno_multi_channels_last(self.similarity, inputs_flatmap, actual_batchsize, 
                               self.fmap_count, self.fmap_shape[0] * self.fmap_shape[1])
      outputs = tf.reshape(outputs_flat, tf.shape(inputs))
      return outputs

    output = smart_cond.smart_cond(training, dropped_inputs,
                                          identity_inputs)
    return output

  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'similarity': self.similarity,
        'seed': self.seed
    }
    base_config = super(Pruno2DMulti, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

