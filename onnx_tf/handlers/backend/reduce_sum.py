import copy

import tensorflow as tf

from onnx_tf.handlers.backend_handler import BackendHandler
from onnx_tf.handlers.handler import onnx_op
from onnx_tf.handlers.handler import tf_func
from .math_mixin import ReductionMixin
from onnx_tf.common.tf_helper import tf_shape


@onnx_op("ReduceSum")
@tf_func(tf.reduce_sum)
class ReduceSum(ReductionMixin, BackendHandler):

  @classmethod
  def version_1(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_11(cls, node, **kwargs):
    return cls._common(node, **kwargs)

  @classmethod
  def version_13(cls, node, **kwargs):
      x = kwargs["tensor_dict"][node.inputs[0]]
      attrs = copy.deepcopy(node.attrs)
      if len(node.inputs) > 1:
        x_shape = tf_shape(x)
        x_rank = len(x_shape)
        axes = kwargs["tensor_dict"][node.inputs[1]]
        axes_shape = tf_shape(axes)
        axes_rank = len(axes_shape)
        print(axes_shape)
        print(axes_rank)
        #import pdb; pdb.set_trace()
        if axes_shape[0] == 0:
          axis = None
        else:
          axis = axes_shape[0]
      else:
        axis = None
      print(axis)
      import pdb; pdb.set_trace()
      attrs["axis"] = axis
      # https://github.com/onnx/onnx/issues/585
      attrs["keepdims"] = attrs.pop("keepdims", 1) == 1
      return [cls.make_tensor_from_onnx_node(node, inputs=[x], attrs=attrs, **kwargs)]

