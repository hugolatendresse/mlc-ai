import torch
import numpy as np
import IPython
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T

N, I, J, K = 16, 128, 128, 128
in_shape = (N,I,J)

@tvm.script.ir_module
class TargetModule:
    @T.prim_func
    def bmm_relu(A: T.Buffer(in_shape, "float32"), B: T.Buffer(in_shape, "float32"), C: T.Buffer(in_shape, "float32")) -> None:
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
        for i0, i1, i2, ax in T.grid(N, I, J, K):
            with T.block("Y"):
                n = T.axis.spatial(N, i0)
                i = T.axis.spatial(I, i1)
                j = T.axis.spatial(J, i2)
                k = T.axis.reduce(K, ax)
                with T.init():
                    Y[n, i, j] = T.float32(0)
                Y[n, i, j] = Y[n, i, j] + A[n, i, k] * B[n, k, j]
        for i0, i1, i2 in T.grid(N, I, J):
            with T.block("C"):
                n = T.axis.spatial(N, i0)
                i = T.axis.spatial(I, i1)
                j = T.axis.spatial(J, i2)
                C[n, i, j] = T.max(Y[n, i, j], T.float32(0))


a = np.random.rand(*in_shape).astype("float32")
n = np.random.rand(*in_shape).astype("float32")
# a = np.ones(in_shape).astype("float32")
# b = np.ones(in_shape).astype("float32")


expected = a @ n



rt_lib = tvm.build(MyBmmRelu, target="llvm")
a_tvm = tvm.nd.array(a)
b_tvm = tvm.nd.array(n)
c_tvm = tvm.nd.array(np.random.rand(16, 128, 128).astype("float32"))
rt_lib["bmm_relu"](a_tvm, b_tvm, c_tvm)

rt_lib_target = tvm.build(TargetModule, target="llvm")
a_tvm2 = tvm.nd.array(a)
b_tvm2 = tvm.nd.array(n)
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
C = sch.get_block("C", func_name="bmm_relu")


# Step 2. Get loops
i0, i1, i2, ax = sch.get_loops(Y)
# nC, iC, jC = sch.get_loops(C)

# Step 3. Organize the loops
kfactor = 4
ax1_0, ax1_1 = sch.split(ax, [None, kfactor])
jfactor = 8
i2_0, ax0_init = sch.split(i2, [None, jfactor])
# sch.reorder(kY_0, kY_1, i2)


sch.reverse_compute_at(block=C, loop=i2_0)
sch.mod.show()

# Step 4. decompose reduction
i0, i1, i2_0, i2_1, ax_0, ax_1 = sch.get_loops(Y)
Y_init = sch.decompose_reduction(block=Y, loop=i2_1)
Y_update = sch.get_block("Y_update", func_name="bmm_relu")
sch.mod.show()

# Step 5. vectorize / parallel / unroll
i0, i1, i2_0, i2_1_init = sch.get_loops(Y_init)
sch.vectorize(loop=i2_1_init)
sch.mod.show()
sch.parallel(loop=i0)
sch.mod.show()
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