Avalanche - OpenCL deep learning framework with a backend for Keras
===================================================================
*Avalanche* is a simple deep learning framework written in C++ and Python.
Unlike the majority of the existing tools it is based on [OpenCL](https://en.wikipedia.org/wiki/OpenCL),
an open computing standard. This allows Avalanche to work on pretty
much any GPU, including the ones made by Intel and AMD, even quite old models.

The project was created as an attempt to better understand how modern deep learning
frameworks like TensorFlow do their job and to practice programming GPUs.
Like any decent deep ML framework these days,
Avalanche is based on a computational graph model. It supports
automatic differentiation, broadcasted operations, automatic memory management,
can utilize multiple GPUs if needed.

The framework also works a backend for Keras, so if you know Keras, you can begin to use
Avalanche without the need to learn anything about it.

Please note, however, that this project is highly experimental and it lacks many
features typical for more mature projects. So if you're new in the field,
better try something like TensorFlow, PyTorch or PlaidML first
(the last one if you need supports of OpenCL).


Installation
------------
First of all, make sure that you have

- [A C++14 compliant compiler](https://en.cppreference.com/w/cpp/compiler_support)
- CMake >= 3.3
- Python >= 3.6.1
- OpenCL libraries / drivers installed

if you use virtualenv for Python, don't forget to activate the sandbox

the rest is simple:

    git clone https://github.com/kpot/avalanche.git
    cd avalanche
    pip install .

to run any Keras code with Avalanche as a backend, you can call it as

    KERAS_BACKEND='avalanche' python my_code.py

or just follow the official
[Keras instruction](https://keras.io/backend/#switching-from-one-backend-to-another)
on how to achieve the same by changing Keras configuration file.

