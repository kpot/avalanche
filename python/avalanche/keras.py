import collections
import threading
from contextlib import contextmanager

import numpy as np

import pyvalanche as av

_thread_local = threading.local()
_contexts = {}
_contexts_lock = threading.Lock()
_uid_prefixes = collections.defaultdict(int)


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
    variable = av.variable(name, value.shape, avalanche_dtype(dtype),
                           av.value_initializer(value))
    variable._uses_learning_phase = False
    variable._keras_shape = value.shape
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
        raise ValueError('The shape must be fully specified')
    if dtype is None:
        dtype = floatx()
    if sparse:
        raise NotImplementedError('Sparse tensors are not supported')
    if name is None:
        name = 'placeholder' + str(get_uid('placeholder'))
    initializer = av.value_initializer(np.zeros(shape))
    p = av.variable(name, shape, avalanche_dtype(dtype), initializer)
    p._uses_learning_phase = False
    p._keras_shape = shape
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
        return results


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
    if hasattr(x, '_keras_shape'):
        return x._keras_shape
    else:
        return None
