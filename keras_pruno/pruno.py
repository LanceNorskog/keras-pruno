import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.framework import smart_cond

import numpy as np

def pruno_random_channels_first(similarity, seed, inputs_flat, actual_batchsize, fmap_count, fmap_size):
    fmap_even = (fmap_count//2)*2
    flatshape = (-1, fmap_count, fmap_size)
    gam = tf.math.reduce_mean(inputs_flat, axis=-1, keepdims=True)
    live = tf.cast(inputs_flat[:, :, :] > gam[:, :, :], dtype='float32', name='live')
    indices = tf.constant(np.arange(fmap_count), dtype='int32')
    random_indices = tf.random.shuffle(indices, seed=seed)
    mult_list = []
    for fm in range(0, fmap_even, 2):
        mult_list.append(live[:, random_indices[fm], :] * live[:, random_indices[fm + 1], :])
    mult = tf.stack(mult_list, axis=1, name='stack1')
    percent = tf.math.reduce_sum(mult, axis=-1, keepdims=True, name='percent') / flatshape[2]
    mask = tf.cast(percent < similarity, dtype='float32', name='mask')
    mask_unstack = tf.unstack(mask, axis=1, name='unstack1')
    mask_list = []
    for i in range(fmap_even // 2):
        mask_list.append(mask_unstack[i])
        mask_list.append(mask_unstack[i])
    if fmap_count > fmap_even:
        mask_list.append(tf.ones_like(mask_unstack[0], dtype='float32'))
    dup_mask = tf.stack(mask_list, axis=1, name='stack2')
    tiled_indices_flat = tf.tile(random_indices, actual_batchsize)
    tiled_indices = tf.reshape(tiled_indices_flat, (-1, flatshape[1], 1))
    inverse_mask_flat = tf.gather(dup_mask, tiled_indices, batch_dims=1, axis=1)
    inverse_mask = tf.reshape(inverse_mask_flat, (-1, fmap_count, 1))
    return inputs_flat * inverse_mask

def pruno_random_channels_last(similarity, seed, inputs_flat, actual_batchsize, fmap_count, fmap_size):
    fmap_even = (fmap_count//2)*2
    flatshape = (-1, fmap_size, fmap_count)
    gam = tf.math.reduce_mean(inputs_flat, axis=1, keepdims=True)
    live = tf.cast(inputs_flat[:, :, :] > gam[:, :, :], dtype='float32')
    indices = tf.constant(np.arange(fmap_count), dtype='int32')
    random_indices = tf.random.shuffle(indices, seed=seed)
    mult_list = []
    for fm in range(0, fmap_even, 2):
        mult_list.append(live[:, :, random_indices[fm]] * live[:, :, random_indices[fm + 1]])
    mult = tf.stack(mult_list, axis=2)
    percent = tf.math.reduce_sum(mult, axis=1, keepdims=True) / flatshape[1]
    mask = tf.cast(percent < similarity, dtype='float32')
    mask_unstack = tf.unstack(mask, axis=2)
    mask_list = []
    for i in range(fmap_even // 2):
        mask_list.append(mask_unstack[i])
        mask_list.append(mask_unstack[i])
    if fmap_even < fmap_count:
        mask_list.append(tf.ones_like(mask_unstack[0], dtype='float32'))
    dup_mask = tf.stack(mask_list, axis=1)
    tiled_indices_flat = tf.tile(random_indices, actual_batchsize)
    tiled_indices = tf.reshape(tiled_indices_flat, (-1, 1, fmap_count))
    inverse_mask_flat = tf.gather(dup_mask, tiled_indices, batch_dims=1, axis=1)
    inverse_mask = tf.reshape(inverse_mask_flat, (-1, 1, fmap_count))
    return inputs_flat * inverse_mask

def pruno_random_channels_batchwise(similarity, seed, inputs_flat, actual_batchsize, fmap_count, fmap_size):
    fmap_even = (fmap_count//2)*2
    flatshape = (-1, fmap_size, fmap_count)
    gam = tf.math.reduce_mean(inputs_flat, axis=1, keepdims=True)
    live = tf.cast(inputs_flat[:, :, :] > gam[:, :, :], dtype='float32')
    indices = tf.constant(np.arange(fmap_count), dtype='int32')
    random_indices = tf.random.shuffle(indices, seed=seed)
    mult_list = []
    for fm in range(0, fmap_even, 2):
        mult_list.append(live[:, :, random_indices[fm]] * live[:, :, random_indices[fm + 1]])
    mult = tf.stack(mult_list, axis=2)
    percent = tf.math.reduce_sum(tf.math.reduce_sum(mult, axis=1), axis=0) / \
        (fmap_size * (tf.cast(actual_batchsize[0], dtype='float32')))
    mask = tf.cast(percent < similarity, dtype='float32')
    mask_unstack = tf.unstack(mask)
    mask_list = []
    for i in range(fmap_even // 2):
        mask_list.append(mask_unstack[i])
        mask_list.append(mask_unstack[i])
    if fmap_even < fmap_count:
        mask_list.append(tf.ones_like(mask_unstack[0], dtype='float32'))
    dup_mask = tf.stack(mask_list)
    ones = tf.ones((actual_batchsize[0], 1, 1), dtype='float32')
    dup_mask = dup_mask * ones
    dup_mask = tf.reshape(dup_mask, (-1, fmap_count, 1))
    tiled_indices_flat = tf.tile(random_indices, actual_batchsize)
    tiled_indices = tf.reshape(tiled_indices_flat, (-1, 1, fmap_count))
    inverse_mask_flat = tf.gather(dup_mask, tiled_indices, batch_dims=1, axis=1)
    inverse_mask = tf.reshape(inverse_mask_flat, (-1, 1, fmap_count))
    return inputs_flat * inverse_mask

class Pruno2D(tf.keras.layers.Layer):
    """Applies Pruning Dropout to the input.
    The Pruno2D layer compares randomly chosen pairs of feature maps, and sets
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
  
    def __init__(self, similarity, batchwise=True, noise_shape=None, seed=None, **kwargs):
        super(Pruno2D, self).__init__(**kwargs)
        if similarity < 0.0 or similarity > 1.0:
            raise ValueError('similarity must be between 0.0 and 1.0: %s' % str(similarity))
        self.similarity = similarity
        self.seed = seed
        self.batchwise = batchwise
        self.supports_masking = True
  
    def build(self, input_shape):
        shape = input_shape.as_list()
        print('build:', input_shape)
        self.fmap_count = shape[3]
        self.fmap_even = (shape[3]//2)*2
        self.fmap_shape = (shape[1], shape[2])
        print(self.fmap_count, self.fmap_even, self.fmap_shape)
        self.built = True
  
    def call(self, inputs, training=None):
        if training is None:
            training = K.learning_phase()
      
        training = True
  
        def identity_inputs():
            return inputs
  
        def dropped_inputs():
            actual_batchsize = tf.shape(inputs)[0:1]
            input_shape = (-1, self.fmap_shape[0], self.fmap_shape[1], self.fmap_count)
            flatshape = (-1, self.fmap_shape[0] * self.fmap_shape[1], self.fmap_count)
            inputs_flatmap = tf.reshape(inputs, flatshape)
            if self.batchwise:
                outputs_flat = pruno_random_channels_batchwise(self.similarity, self.seed, inputs_flatmap, actual_batchsize, 
                                 self.fmap_count, self.fmap_shape[0] * self.fmap_shape[1])
            else:
                outputs_flat = pruno_random_channels_last(self.similarity, self.seed, inputs_flatmap, actual_batchsize, 
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
            'batchwise': self.batchwise,
            'seed': self.seed
        }
        base_config = super(Pruno2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
