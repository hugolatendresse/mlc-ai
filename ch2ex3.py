import torch
import numpy as np
import IPython
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T

N, I, J, K = 16, 128, 128, 128

@tvm.script.ir_module
class TargetModule:
    @T.prim_func
    def bmm_relu(A: T.Buffer((16, 128, 128), "float32"), B: T.Buffer((16, 128, 128), "float32"), C: T.Buffer((16, 128, 128), "float32")) -> None:
        T.func_attr({"global_symbol": "bmm_relu", "tir.noalias": True})
        Y = T.alloc_buffer([16, 128, 128], dtype="float32")
        for i0 in T.parallel(16):
            for i1, i2_0 in T.grid(128, 16):
                for ax0_init in T.vectorized(8):
                    with T.block("Y_init"):
                        n, i = T.axis.remap("SS", [i0, i1])
                        j = T.axis.spatial(128, i2_0 * 8 + ax0_init)
                        Y[n, i, j] = T.float32(0)
                for ax1_0 in T.serial(32):
                    for ax1_1 in T.unroll(4):
                        for ax0 in T.serial(8):
                            with T.block("Y_update"):
                                n, i = T.axis.remap("SS", [i0, i1])
                                j = T.axis.spatial(128, i2_0 * 8 + ax0)
                                k = T.axis.reduce(128, ax1_0 * 4 + ax1_1)
                                Y[n, i, j] = Y[n, i, j] + A[n, i, k] * B[n, k, j]
                for i2_1 in T.vectorized(8):
                    with T.block("C"):
                        n, i = T.axis.remap("SS", [i0, i1])
                        j = T.axis.spatial(128, i2_0 * 8 + i2_1)
                        C[n, i, j] = T.max(Y[n, i, j], T.float32(0))

@tvm.script.ir_module
class MyBmmRelu:
    @T.prim_func
    def bmm_relu(A : T.buffer(shape=(N, I, J), dtype="float32"),
                 B : T.buffer(shape=(N, I, J), dtype="float32"),
                 C : T.buffer(shape=(N, I, J), dtype="float32")):
        T.func_attr({"global_symbol": "bmm_relu", "tir.noalias": True})
        Y = T.alloc_buffer((N, I, J), dtype="float32")
        for n, i, j, k in T.grid(N, I, J, K):
            with T.block("Y"):
                vn = T.axis.spatial(N, n)
                vi = T.axis.spatial(I, i)
                vj = T.axis.spatial(J, j)
                vk = T.axis.reduce(K, k)
                with T.init():
                    Y[vn, vi, vj] = T.float32(0)
                Y[vn, vi, vj] = Y[vn, vi, vj] + A[vn, vi, vk] * B[vn, vk, vj]
        for n, i, j in T.grid(N, I, J):
            with T.block("C"):
                vn = T.axis.spatial(N, n)
                vi = T.axis.spatial(I, i)
                vj = T.axis.spatial(J, j)
                C[vn, vi, vj] = T.max(Y[vn, vi, vj], T.float32(0))

in_shape = (N,I,J)

a = np.random.rand(*in_shape).astype("float32")
b = np.random.rand(*in_shape).astype("float32")
# a = np.ones(in_shape).astype("float32")
# b = np.ones(in_shape).astype("float32")


expected = a @ b



rt_lib = tvm.build(MyBmmRelu, target="llvm")
a_tvm = tvm.nd.array(a)
b_tvm = tvm.nd.array(b)
c_tvm = tvm.nd.array(np.random.rand(16, 128, 128).astype("float32"))
rt_lib["bmm_relu"](a_tvm, b_tvm, c_tvm)

rt_lib_target = tvm.build(TargetModule, target="llvm")
a_tvm2 = tvm.nd.array(a)
b_tvm2 = tvm.nd.array(b)
c_tvm_target = tvm.nd.array(np.random.rand(16, 128, 128).astype("float32"))
rt_lib_target["bmm_relu"](a_tvm2, b_tvm2, c_tvm_target)

actual_tvm = c_tvm.numpy()
expected_tvm = c_tvm_target.numpy()
np.testing.assert_allclose(actual_tvm, expected_tvm, rtol=1e-5)

sch = tvm.tir.Schedule(MyBmmRelu)
# Hints: you can use
# `IPython.display.Code(sch.mod.script(), language="python")`
# or `print(sch.mod.script())`
# to show the current program at any time during the transformation.

# Step 1. Get blocks
Y = sch.get_block("Y", func_name="bmm_relu")
...

# Step 2. Get loops
b, i, j, k = sch.get_loops(Y)
...

# Step 3. Organize the loops
k0, k1 = sch.split(k, ...)
sch.reorder(...)
sch.compute_at/reverse_compute_at(...)
...

# Step 4. decompose reduction
Y_init = sch.decompose_reduction(Y, ...)
...

# Step 5. vectorize / parallel / unroll
sch.vectorize(...)
sch.parallel(...)
sch.unroll(...)
...

IPython.display.Code(sch.mod.script(), language="python")

tvm.ir.assert_structural_equal(sch.mod, TargetModule)
print("Pass")

before_rt_lib = tvm.build(MyBmmRelu, target="llvm")
after_rt_lib = tvm.build(sch.mod, target="llvm")
a_tvm = tvm.nd.array(np.random.rand(16, 128, 128).astype("float32"))
b_tvm = tvm.nd.array(np.random.rand(16, 128, 128).astype("float32"))
c_tvm = tvm.nd.array(np.random.rand(16, 128, 128).astype("float32"))
after_rt_lib["bmm_relu"](a_tvm, b_tvm, c_tvm)
before_timer = before_rt_lib.time_evaluator("bmm_relu", tvm.cpu())
print("Before transformation:")
print(before_timer(a_tvm, b_tvm, c_tvm))

f_timer = after_rt_lib.time_evaluator("bmm_relu", tvm.cpu())
print("After transformation:")
print(f_timer(a_tvm, b_tvm, c_tvm))