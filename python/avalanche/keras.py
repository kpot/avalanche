import collections
import threading
from contextlib import contextmanager

import numpy as np

import pyvalanche as av

_thread_local = threading.local()
_contexts = {}
_contexts_lock = threading.Lock()
_uid_prefixes = collections.defaultdict(int)


_IMAGE_DATA_FORMAT = 'channels_last'


def floatx():
    from keras.backend.common import floatx as keras_floatx
    return keras_floatx()


def get_context(device_id: int=None) -> av.Context:
    """
    Returns computational context (similar to tf.Session) for any currently
    activated (via `use_device`) device.
    Contexts get cached so any subsequent call will return the same context
    for the device.
    :param device_id: a device id for which the context should be created.
        If None, the first available device will be selected, or the one
        chosen by `use_device`.
    """
    with _contexts_lock:
        if device_id is None:
            try:
                device_id = getattr(_thread_local, 'device_id')
            except AttributeError:
                manager = av.default_memory_manager()
                if manager.num_devices == 0:
                    raise RuntimeError(
                        "Avalanche could not find any devices "
                        "suitable for computation")
                _thread_local.device_id = device_id = 0
                info = manager.device_info(device_id)
                print('Avalanche is going to use device {} ({}) by default. '
                      'Total devices available: {}'
                      .format(device_id, info.name, manager.num_devices))
        try:
            return _contexts[device_id]
        except KeyError:
            context = av.Context.make_for_device(device_id)
            _contexts[device_id] = context
            return context


class use_device:
    def __init__(self, device_id: int):
        self.device_id = device_id
        self.prev_device_id = None

    def __enter__(self):
        self.prev_device_id = getattr(_thread_local, 'device_id', None)
        _thread_local.device_id = self.device_id

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.prev_device_id is None:
            del _thread_local.device_id
        else:
            _thread_local.device_id = self.prev_device_id


def avalanche_dtype(dtype):
    if isinstance(dtype, str):
        if not hasattr(av.ArrayType, dtype):
            dtype = np.dtype(dtype).name
        return getattr(av.ArrayType, dtype)
    elif isinstance(dtype, np.dtype):
        return getattr(av.ArrayType, dtype.name)
    elif callable(dtype):
        sample_value = dtype()
        dtype = sample_value.dtype.name
        return getattr(av.ArrayType, dtype.name)
    else:
        raise ValueError('Unknown dtype {!r}'.format(dtype))


def variable(value, dtype=None, name=None, constraint=None):
    """Instantiates a variable and returns it.

    # Arguments
        value: Numpy array, initial value of the tensor.
        dtype: Tensor type.
        name: Optional name string for the tensor.
        constraint: Optional projection function to be
            applied to the variable after an optimizer update.

    # Returns
        A variable instance (with Keras metadata included).

    # Examples
    ```python
        >>> from keras import backend as K
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val, dtype='float64', name='example_var')
        >>> K.dtype(kvar)
        'float64'
        >>> print(kvar)
        example_var
        >>> K.eval(kvar)
        array([[ 1.,  2.],
               [ 3.,  4.]])
    ```
    """
    if name is None:
        name = 'variable' + str(get_uid('variable'))
    if dtype is None:
        dtype = floatx()
    if constraint is not None:
        raise NotImplementedError('Constraints are not supported')
    if is_tensor(value):
        variable = av.variable_from_node(name, value)
    else:
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        variable = av.variable(
            name, value.shape, avalanche_dtype(dtype),
            av.value_initializer(value))
    variable._uses_learning_phase = False
    variable._keras_shape = value.shape
    variable._is_variable = True
    return variable


def placeholder(shape=None, ndim=None, dtype=None, sparse=False, name=None):
    """Instantiates a placeholder tensor and returns it.

    # Arguments
        shape: Shape of the placeholder
            (integer tuple, may include `None` entries).
        ndim: Number of axes of the tensor.
            At least one of {`shape`, `ndim`} must be specified.
            If both are specified, `shape` is used.
        dtype: Placeholder type.
        sparse: Boolean, whether the placeholder should have a sparse type.
        name: Optional name string for the placeholder.

    # Returns
        Tensor instance (with Keras metadata included).

    # Examples
    ```python
        >>> from keras import backend as K
        >>> input_ph = K.placeholder(shape=(2, 4, 5))
        >>> input_ph._keras_shape
        (2, 4, 5)
        >>> input_ph
        <tf.Tensor 'Placeholder_4:0' shape=(2, 4, 5) dtype=float32>
    ```
    """
    if shape is None:
        if ndim is None:
            raise ValueError('Either the shape of ndim must be provided')
        else:
            shape = [-1] * ndim
    if dtype is None:
        dtype = floatx()
    if sparse:
        raise NotImplementedError('Sparse tensors are not supported')
    if name is None:
        name = 'placeholder' + str(get_uid('placeholder'))
    p = av.placeholder(name, shape, avalanche_dtype(dtype))
    p._uses_learning_phase = False
    p._keras_shape = shape
    p._is_placeholder = True
    return p


def function(inputs, outputs, updates=None, **kwargs):
    """Instantiates a Keras function.

    # Arguments
        inputs: List of placeholder tensors.
        outputs: List of output tensors.
        updates: List of update ops.
        **kwargs: currently ignored

    # Returns
        Output values as Numpy arrays.

    # Raises
        ValueError: if invalid kwargs are passed in.
    """
    return Function(inputs, outputs, updates=updates, **kwargs)


class Function(object):
    """Runs a computation graph.

    # Arguments
        inputs: Feed placeholders to the computation graph.
        outputs: Output tensors to fetch.
        updates: Additional update ops to be run at function call.
        name: a name to help users identify what this function does.
        session_kwargs: arguments to `tf.Session.run()`:
            `fetches`, `feed_dict`,
            `options`, `run_metadata`
    """
    def __init__(self, inputs, outputs, updates=None,
                 name=None,
                 **session_kwargs):
        updates = updates or []
        if not isinstance(inputs, (list, tuple)):
            raise TypeError('`inputs` to a the backend function '
                            'should be a list or tuple.')
        if not isinstance(outputs, (list, tuple)):
            raise TypeError('`outputs` of a the backend function '
                            'should be a list or tuple.')
        if not isinstance(updates, (list, tuple)):
            raise TypeError('`updates` in a the backend function '
                            'should be a list or tuple.')
        self.name = name
        self.context = get_context()
        self.placeholders = inputs
        self.executor = av.Executor(self.context, outputs)

    def __call__(self, inputs):
        for node, value in zip(self.placeholders, inputs):
            self.context.init(node, value)
        results = self.executor.run()
        return [r.asnumpy() for r in results]


def get_uid(prefix=''):
    """Provides a unique UID given a string prefix.

    # Arguments
        prefix: string.

    # Returns
        An integer.

    # Example
    ```python
        >>> keras.backend.get_uid('dense')
        1
        >>> keras.backend.get_uid('dense')
        2
    ```

    """
    _uid_prefixes[prefix] += 1
    return _uid_prefixes[prefix]


NAME_SCOPE_STACK = []

@contextmanager
def name_scope(name):
    global NAME_SCOPE_STACK
    NAME_SCOPE_STACK.append(name)
    yield
    NAME_SCOPE_STACK.pop()


def is_keras_tensor(x):
    """Returns whether `x` is a Keras tensor.

    A "Keras tensor" is a tensor that was returned by a Keras layer,
    (`Layer` class) or by `Input`.

    # Arguments
        x: A candidate tensor.

    # Returns
        A boolean: Whether the argument is a Keras tensor.

    # Raises
        ValueError: In case `x` is not a symbolic tensor.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> from keras.layers import Input, Dense
        >>> np_var = numpy.array([1, 2])
        >>> K.is_keras_tensor(np_var) # A numpy array is not a symbolic tensor.
        ValueError
        >>> k_var = tf.placeholder('float32', shape=(1,1))
        >>> K.is_keras_tensor(k_var) # A variable indirectly created outside of keras is not a Keras tensor.
        False
        >>> keras_var = K.variable(np_var)
        >>> K.is_keras_tensor(keras_var)  # A variable created with the keras backend is not a Keras tensor.
        False
        >>> keras_placeholder = K.placeholder(shape=(2, 4, 5))
        >>> K.is_keras_tensor(keras_placeholder)  # A placeholder is not a Keras tensor.
        False
        >>> keras_input = Input([10])
        >>> K.is_keras_tensor(keras_input) # An Input is a Keras tensor.
        True
        >>> keras_layer_output = Dense(10)(keras_input)
        >>> K.is_keras_tensor(keras_layer_output) # Any Keras layer output is a Keras tensor.
        True
    ```
    """
    if not is_tensor(x):
        raise ValueError('Unexpectedly found an instance of type `' +
                         str(type(x)) + '`. '
                                        'Expected a symbolic tensor instance.')
    return hasattr(x, '_keras_history')


def is_tensor(x):
    return isinstance(x, av.BaseNode)


def ndim(x):
    """Returns the number of axes in a tensor, as an integer.

    # Arguments
        x: Tensor or variable.

    # Returns
        Integer (scalar), number of axes.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> inputs = K.placeholder(shape=(2, 4, 5))
        >>> val = np.array([[1, 2], [3, 4]])
        >>> kvar = K.variable(value=val)
        >>> K.ndim(inputs)
        3
        >>> K.ndim(kvar)
        2
    ```
    """
    return x.shape.rank


def int_shape(x):
    """Returns the shape of a Keras tensor or a Keras variable as a tuple of
    integers or None entries.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tuple of integers (or None entries).
    """
    # raise NotImplementedError('Adjust with respect to non-full shapes')
    if hasattr(x, '_keras_shape'):
        return x._keras_shape
    elif hasattr(x, 'shape'):
        return [(i if i >= 0 else None) for i in x.shape]
    else:
        return None


def random_uniform(shape, minval=0.0, maxval=1.0, dtype=None, seed=None):
    """Returns a tensor with uniform distribution of values.

    # Arguments
        shape: A tuple of integers, the shape of tensor to create.
        minval: A float, lower boundary of the uniform distribution
            to draw samples.
        maxval: A float, upper boundary of the uniform distribution
            to draw samples.
        dtype: String, dtype of returned tensor.
        seed: Integer, random seed.

    # Returns
        A tensor.
    """
    if dtype is None:
        dtype = floatx()
    if seed is None:
        seed = np.random.randint(10e6)
    return av.random_uniform(av.Shape(shape), minval, maxval, avalanche_dtype(dtype), seed);


def constant(value, dtype=None, shape=None, name=None):
    """Creates a constant tensor.

    # Arguments
        value: A constant value (or list)
        dtype: The type of the elements of the resulting tensor.
        shape: Optional dimensions of resulting tensor.
        name: Optional name for the tensor.

    # Returns
        A Constant Tensor.
    """
    if dtype is None:
        dtype = floatx()
    if not is_tensor(value):
        value = np.array(value, dtype=dtype)
    if shape is not None:
        value = np.full(shape, value, dtype=dtype)
    if name is None:
        name = 'constant' + str(get_uid('constant'))
    return av.constant(value, name)


def dot(x, y):
    """Multiplies 2 tensors (and/or variables) and returns a *tensor*.

    When attempting to multiply a nD tensor
    with a nD tensor, it reproduces the Theano behavior.
    (e.g. `(2, 3) * (4, 3, 5) -> (2, 4, 5)`)

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A tensor, dot product of `x` and `y`.

    # Examples
    ```python
        # dot product between tensors
        >>> x = K.placeholder(shape=(2, 3))
        >>> y = K.placeholder(shape=(3, 4))
        >>> xy = K.dot(x, y)
        >>> xy
        <tf.Tensor 'MatMul_9:0' shape=(2, 4) dtype=float32>
    ```

    ```python
        # dot product between tensors
        >>> x = K.placeholder(shape=(32, 28, 3))
        >>> y = K.placeholder(shape=(3, 4))
        >>> xy = K.dot(x, y)
        >>> xy
        <tf.Tensor 'MatMul_9:0' shape=(32, 28, 4) dtype=float32>
    ```

    ```python
        # Theano-like behavior example
        >>> x = K.random_uniform_variable(shape=(2, 3), low=0, high=1)
        >>> y = K.ones((4, 3, 5))
        >>> xy = K.dot(x, y)
        >>> K.int_shape(xy)
        (2, 4, 5)
    ```
    """
    if ndim(x) != 2 and ndim(y) != 2:
        raise ValueError(
            'dot product is currently implemented only for 2D tensors')
    out = av.ops.matmul(x, y)
    return out


def image_data_format():
    """Returns the default image data format convention ('channels_first' or 'channels_last').

    # Returns
        A string, either `'channels_first'` or `'channels_last'`

    # Example
    ```python
        >>> keras.backend.image_data_format()
        'channels_first'
    ```
    """
    return _IMAGE_DATA_FORMAT


def bias_add(x, bias, data_format=None):
    """Adds a bias vector to a tensor.

    # Arguments
        x: Tensor or variable.
        bias: Bias tensor to add.
        data_format: string, `"channels_last"` or `"channels_first"`.

    # Returns
        Output tensor.

    # Raises
        ValueError: In one of the two cases below:
                    1. invalid `data_format` argument.
                    2. invalid bias shape.
                       the bias should be either a vector or
                       a tensor with ndim(x) - 1 dimension
    """
    if data_format is None:
        data_format = image_data_format()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('Unknown data_format: ' + str(data_format))
    if data_format == 'channels_last':
        return av.ops.plus(x, bias)
    else:
        bias_shape = int_shape(bias)
        dims = [1] * ndim(x)
        dims[:ndim(bias)] = bias_shape
        return av.ops.plus(x, av.reshape(bias, av.Shape(dims)))


def relu(x, alpha=0., max_value=None):
    """Rectified linear unit.

    With default values, it returns element-wise `max(x, 0)`.

    # Arguments
        x: A tensor or variable.
        alpha: A scalar, slope of negative section (default=`0.`).
        max_value: Saturation threshold.

    # Returns
        A tensor.
    """
    if alpha != 0.:
        raise NotImplementedError(
            "Leaky ReLU has not been implemented in avalanche yet")
    else:
        x = av.ops.relu(x)

    if max_value is not None:
        raise NotImplementedError(
            "Saturation threashold for ReLU has not "
            "been implemented in avalanche yet")
    return x


def sigmoid(x):
    """Element-wise sigmoid.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    """
    return av.ops.sigmoid(x)


def tanh(x):
    """Element-wise tanh.

    # Arguments
        x: A tensor or variable.

    # Returns
        A tensor.
    """
    return av.ops.tanh(x)


def is_sparse(tensor):
    """Returns whether a tensor is a sparse tensor.

    # Arguments
        tensor: A tensor instance.

    # Returns
        A boolean.

    # Example
    ```python
        >>> from keras import backend as K
        >>> a = K.placeholder((2, 2), sparse=False)
        >>> print(K.is_sparse(a))
        False
        >>> b = K.placeholder((2, 2), sparse=True)
        >>> print(K.is_sparse(b))
        True
    ```
    """
    return False


def dtype(x):
    """Returns the dtype of a Keras tensor or variable, as a string.

    # Arguments
        x: Tensor or variable.

    # Returns
        String, dtype of `x`.

    # Examples
    ```python
        >>> from keras import backend as K
        >>> K.dtype(K.placeholder(shape=(2,4,5)))
        'float32'
        >>> K.dtype(K.placeholder(shape=(2,4,5), dtype='float32'))
        'float32'
        >>> K.dtype(K.placeholder(shape=(2,4,5), dtype='float64'))
        'float64'
        # Keras variable
        >>> kvar = K.variable(np.array([[1, 2], [3, 4]]))
        >>> K.dtype(kvar)
        'float32_ref'
        >>> kvar = K.variable(np.array([[1, 2], [3, 4]]), dtype='float32')
        >>> K.dtype(kvar)
        'float32_ref'
    ```
    """
    prefix = 'ref_' if getattr(x, '_is_variable', False) else ''
    return prefix + x.dtype.name


def is_placeholder(x):
    """Returns whether `x` is a placeholder.

    # Arguments
        x: A candidate placeholder.

    # Returns
        Boolean.
    """
    return getattr(x, '_is_placeholder', False)


def mean(x, axis=None, keepdims=False):
    """Mean of a tensor, alongside the specified axis.

    # Arguments
        x: A tensor or variable.
        axis: A list of integer. Axes to compute the mean.
        keepdims: A boolean, whether to keep the dimensions or not.
            If `keepdims` is `False`, the rank of the tensor is reduced
            by 1 for each entry in `axis`. If `keepdims` is `True`,
            the reduced dimensions are retained with length 1.

    # Returns
        A tensor with the mean of elements of `x`.
    """
    if isinstance(axis, int):
        axis = [axis]
    elif axis is None:
        axis = []
    return av.ops.reduce_mean(x, axis, keepdims)


def square(x):
    """Element-wise square.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    return av.ops.square(x)


def cast(x, dtype):
    """Casts a tensor to a different dtype and returns it.

    You can cast a Keras variable but it still returns a Keras tensor.

    # Arguments
        x: Keras tensor (or variable).
        dtype: String, either (`'float16'`, `'float32'`, or `'float64'`).

    # Returns
        Keras tensor with dtype `dtype`.

    # Example
    ```python
        >>> from keras import backend as K
        >>> input = K.placeholder((2, 3), dtype='float32')
        >>> input
        <tf.Tensor 'Placeholder_2:0' shape=(2, 3) dtype=float32>
        # It doesn't work in-place as below.
        >>> K.cast(input, dtype='float16')
        <tf.Tensor 'Cast_1:0' shape=(2, 3) dtype=float16>
        >>> input
        <tf.Tensor 'Placeholder_2:0' shape=(2, 3) dtype=float32>
        # you need to assign it.
        >>> input = K.cast(input, dtype='float16')
        >>> input
        <tf.Tensor 'Cast_2:0' shape=(2, 3) dtype=float16>
    ```
    """
    return av.ops.cast(x, avalanche_dtype(dtype))


def equal(x, y):
    """Element-wise equality between two tensors.

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.
    """
    return av.ops.equal(x, y)


def not_equal(x, y):
    """Element-wise inequality between two tensors.

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.
    """
    return av.ops.not_equal(x, y)


def greater(x, y):
    """Element-wise truth value of (x > y).

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.
    """
    return av.ops.greater(x, y)


def greater_equal(x, y):
    """Element-wise truth value of (x >= y).

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.
    """
    return av.ops.greater_equal(x, y)


def less(x, y):
    """Element-wise truth value of (x < y).

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.
    """
    return av.ops.less(x, y)


def less_equal(x, y):
    """Element-wise truth value of (x <= y).

    # Arguments
        x: Tensor or variable.
        y: Tensor or variable.

    # Returns
        A bool tensor.
    """
    return av.ops.less_equal(x, y)


def gradients(loss, variables):
    """Returns the gradients of `loss` w.r.t. `variables`.

    # Arguments
        loss: Scalar tensor to minimize.
        variables: List of variables.

    # Returns
        A gradients tensor.
    """
    return av.gradients(loss, variables)


def update_add(x, increment):
    """Update the value of `x` by adding `increment`.

    # Arguments
        x: A `Variable`.
        increment: A tensor of same shape as `x`.

    # Returns
        The variable `x` updated.
    """
    return av.ops.update_add(x, increment)


def update_sub(x, decrement):
    """Update the value of `x` by subtracting `decrement`.

    # Arguments
        x: A `Variable`.
        decrement: A tensor of same shape as `x`.

    # Returns
        The variable `x` updated.
    """
    return av.ops.update_sub(x, decrement)


def sqrt(x):
    """Element-wise square root.

    # Arguments
        x: Tensor or variable.

    # Returns
        A tensor.
    """
    # zero = _to_tensor(0., x.dtype.base_dtype)
    # inf = _to_tensor(np.inf, x.dtype.base_dtype)
    # x = tf.clip_by_value(x, zero, inf)
    return av.ops.sqrt(x)


def pow(x, a):
    """Element-wise exponentiation.

    # Arguments
        x: Tensor or variable.
        a: Python integer.

    # Returns
        A tensor.
    """
    if isinstance(a, (int, float)):
        return av.ops.scale_pow(x, 1, a)
    else:
        return av.ops.pow(x, a)

