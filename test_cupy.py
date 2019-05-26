import numpy as np
import cupy as cp

import contextlib
import time

import argparse

print("define and move array, device and host")
a_gpu = cp.array([1, 2, 3]) # define array a_gpu on the current device
b_cpu = np.array([4, 5, 6]) # define array b_cpu on the host
b_gpu = cp.asarray(b_cpu) # move the array to the current device
a_cpu = cp.asnumpy(a_gpu) # move the array to the host
a_cpu = a_gpu.get() # an alternative for the previous line
print(a_cpu)
print(b_cpu)


print("CPU/GPU agnostic method")
def multiply(a, b):
    xp_a = cp.get_array_module(a)
    xp_b = cp.get_array_module(b)
    return a * b
print(multiply(a_gpu, b_gpu)) # call multiply by device arguments
print(multiply(a_cpu, b_cpu)) # call multiply by host arguments


# User-defined Kernels: elementwise, reduction, raw
x = cp.arange(10, dtype=np.float32).reshape(2, 5)
y = cp.arange(5, dtype=np.float32)
z = cp.zeros((2,5), dtype=np.float32)


print('ElementwiseKernel')
squared_diff = cp.ElementwiseKernel(
        # n, i, and names starting with an underscore, '_' are reserved
        'float32 x, float32 y',
        'float32 z',
        'z = (x - y) * (x - y)',
        'squared_diff')
print(squared_diff(x, y))
# Output args can be explicitly specified (next to the input args)
squared_diff(x, y, z)
print(z)
print(squared_diff(x, 5)) # call elementwise method on scalar


# If a type specifier is one character, then it is treated as a "type placeholder". It can be used to define a "type-generic" kernels.
print('ElementwiseKernel type-generic')
squared_diff_generic = cp.ElementwiseKernel(
        'T x, T y',
        'T z',
        'z = (x - y) * (x - y)',
        'squared_diff_generic')
# The ElementwiseKernel class first checks the output arguments and then the input arguments to determine the actual type. If no output arguments are given on the kernel invocation, then only the input arguments are used to determine the type.
print(squared_diff_generic(x, y))


print('ElementwiseKernel type-generic type placeholder in loop body code')
# The type placeholder can be used in the loop body code:
squared_diff_typegeneric = cp.ElementwiseKernel(
        'T x, T y',
        'T z',
        '''
            T diff = x - y;
            z = diff * diff;
        '''
        'squared_diff_generic')
print(squared_diff_generic(x, y))


# More than one type placeholder can be used in a kernel definition.
print('ElementwiseKernel super_type-generic')
squared_diff_super_generic = cp.ElementwiseKernel(
        'X x, Y y',
        'Z z',
        'z = (x - y) * (x - y)',
        'squared_diff')
print(squared_diff_super_generic(x, y, z))
# ?->  Note that this kernel requires the output argument explicitly specified, because the type Z cannot be automatically determined from the input arguments.


print('ElementwiseKernel type-generic manual indexing')
#The ElementwiseKernel class does the indexing with broadcasting automatically, which is useful to define most elementwise computations. On the other hand, we sometimes want to write a kernel with "manual indexing" for some arguments. We can tell the ElementwiseKernel class to use manual indexing by adding the "raw" keyword preceding the type specifier.
add_reverse = cp.ElementwiseKernel( # z = x + y[::-1]
        'T x, raw T y',
        'T z',
        'z = x + y[_ind.size() - i - 1]',
        'add_reverse')
print(add_reverse(x, x))
# i indicates the index within the loop. _ind.size() indicates total number of elements to apply the elementwise operation. A raw argument can be used like an array.
# ?-> Note that raw arguments are not involved in the broadcasting. If you want to mark all arguments as raw, you must specify the size argument on invocation, which defnes the value of _ind.size().


print('ReductionKernel')
# Reduction kernels can be defined by the ReductionKernel class. We can use it by defining four parts of the kernel code:
# 1-> Identity value: This value is used for the initial value of reduction.
# 2-> Mapping expression: It is used for the pre-processing of each element to be reduced.
# 3-> Reduction expression: It is an operator to reduce the multiple mapped values. The special variables a and b are used for its operands.
# 4-> Post mapping expression: It is used to transform the resulting reduced values. The special variable a is used as its input. Output should be written to the output parameter.
# ReductionKernel class automatically inserts other code fragments that are required for an efficient and flexible reduction implementation.
#For example, L2 norm along specified axes can be written as follows:
l2norm_kernel = cp.ReductionKernel(
        'T x', # input params
        'T y', # output params
        'x * x', # map
        'a + b', # reduce
        'y = sqrt(a)', # post-reduction map
        '0', # identity value
        'l2norm' # kernel name
)
print(l2norm_kernel(x, axis=0))
print(l2norm_kernel(x, axis=1))
# "raw" specifier is "restricted" for usages that the axes to be reduced are put at the head of the shape. ?-> It means, if you want to use raw specifier for at least one argument, the axis argument must be 0 or a contiguous increasing sequence of integers starting from 0, like (0, 1), (0, 1, 2), etc.
# RawKernel object allows you to call the kernel with CUDA's cuLaunchKernel interface. In other words, you have control over grid size, block size, shared memory size and stream.
print('RawKernel')
add_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void my_add(const float* x1, const float* x2, float* y) {
        int tid = blockDim.x * blockIdx.x + threadIdx.x;
        y[tid] = x1[tid] + x2[tid];
    }
    ''', 'my_add')
x1 = cp.arange(25, dtype=cp.float32).reshape(5, 5)
x2 = cp.arange(25, dtype=cp.float32).reshape(5, 5)
yy = cp.zeros((5, 5), dtype=cp.float32)
add_kernel((5,), (5,), (x1, x2, yy)) # grid, block and arguments
print(yy)
# The kernel does not have return values. You need to pass both input arrays and output arrays as arguments.
# No validation will be performed by CuPy for arguments passed to the kernel, including types and number of arguments. Especially note that when passing ndarray, its dtype should match with the type of the argument declared in the method signature of the CUDA source code (unless you are casting arrays intentionally). For example, cupy.float32 and cupy.uint64 arrays must be passed to the argument typed as float* and unsigned long long*. For Python primitive types, int, float and bool map to long long, double and bool, respectively.
# When using printf() in your CUDA kernel, you may need to synchronize the stream to see the output. You can use cupy.cuda.Stream.null.synchronize() if you are using the default stream.

print("cupy.fuse() decorator")
# cupy.fuse() is a decorator that fuses functions. This decorator can be used to define an elementwise or reduction kernel more easily than ElementwiseKernel or ReductionKernel.
x_cp = cp.arange(10)
y_cp = cp.arange(10)[::-1]
x_np = np.arange(10)
y_np = np.arange(10)[::-1]


print("Kernel Fusion elementwise")
@cp.fuse()
def squared_diff_fuse(x, y):
    return (x - y) * (x - y)
print(squared_diff_fuse(x_cp, y_cp)) # called on cupy arrays
print(squared_diff_fuse(x_np, y_np)) # called on numpy arrays
print(squared_diff_fuse(x_np, 1)) # called on numpy array and scalar


print("Kernel Fusion reduction")
@cp.fuse()
def sum_of_products(x, y):
    return cp.sum(x * y, axis = -1)
print(sum_of_products(x_cp, y_cp))


# Profiling

@contextlib.contextmanager
def timer(message):
        cp.cuda.Stream.null.synchronize()
        start = time.time()
        yield
        cp.cuda.Stream.null.synchronize()
        end = time.time()
        print('%s:\t%f sec' % (message, end - start))

with timer('timer works! hooray'):
        aaa = cp.zeros(1000000, dtype = cp.float32)
        bbb = aaa + 1


'''
from cupy import prof
with cp.prof.time_range('some range in green', color_id=0): # A context manager to describe the enclosed block as a nested range
        # do something you want to measure
        pass
@cupy.prof.TimeRangeDecorator() # docorated function calls are marked as ranges in NVIDIA profiler timeline.
def function_to_profile():
        # do something you want to measure
        pass
with cp.cuda.profile():
        # do something you want to measure
        pass

cp.cuda.profiler.initialize(unicode config_file, unicode output_file, int output_mode)
cp.cuda.profiler.start() # Enable profiling
cp.cuda.profiler.stop() # Disable profiling
cp.cuda.nvtx.Mark(message, int id_color=-1) # Marks an instantaneous event (marker) in the application.
cp.cuda.nvtx.MarkC(message, int id_color=-1) # Marks an instantaneous event (marker) in the application.
cp.cuda.nvtx.RangePush(message, int id_color=-1) # Starts a nested range.
cp.cuda.nvtx.RangePushC(message, int id_color=-1) # Starts a nested range.
cp.cuda.nvtx.RangePop(message, int id_color=-1) # Ends a nested range.

'''

# Reference Manual at:
# https://docs-cupy.chainer.org/en/stable/reference/index.html