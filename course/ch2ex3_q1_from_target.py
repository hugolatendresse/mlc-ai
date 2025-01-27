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
    def bmm_relu_target(A: T.Buffer(in_shape, "float32"), B: T.Buffer(in_shape, "float32"), C: T.Buffer(in_shape, "float32")) -> None:
        T.func_attr({"global_symbol": "bmm_relu_target", "tir.noalias": True})
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
class AlmostTargetModule:
    @T.prim_func
    def bmm_relu_almost(A: T.Buffer(in_shape, "float32"), B: T.Buffer(in_shape, "float32"), C: T.Buffer(in_shape, "float32")) -> None:
        T.func_attr({"global_symbol": "bmm_relu_almost", "tir.noalias": True})
        Y = T.alloc_buffer([16, 128, 128], dtype="float32")
        for i0 in T.grid(16):
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

a = np.random.rand(*in_shape).astype("float32")
n = np.random.rand(*in_shape).astype("float32")

expected = a @ n

rt_lib = tvm.build(AlmostTargetModule, target="llvm")
a_tvm = tvm.nd.array(a)
b_tvm = tvm.nd.array(n)
c_tvm = tvm.nd.array(np.random.rand(16, 128, 128).astype("float32"))
rt_lib["bmm_relu_almost"](a_tvm, b_tvm, c_tvm)

rt_lib_target = tvm.build(TargetModule, target="llvm")
a_tvm2 = tvm.nd.array(a)
b_tvm2 = tvm.nd.array(n)
c_tvm_target = tvm.nd.array(np.random.rand(16, 128, 128).astype("float32"))
rt_lib_target["bmm_relu_target"](a_tvm2, b_tvm2, c_tvm_target)

actual_tvm = c_tvm.numpy()
expected_tvm = c_tvm_target.numpy()
np.testing.assert_allclose(actual_tvm, expected_tvm, rtol=1e-5)

sch = tvm.tir.Schedule(AlmostTargetModule)

# TODO all three attemps below to parallelize by i0 don't work

Y_init = sch.get_block("Y_init", func_name="bmm_relu_almost")
i0, i1, i2_0, ax0_init = sch.get_loops(Y_init)
sch.parallel(loop=i0) # TODO doesn't work

Y_update = sch.get_block("Y_update", func_name="bmm_relu_almost")
i0, i1, i2_0, ax1_0, ax1_1, ax0 = sch.get_loops(Y_update)
sch.parallel(loop=i0) # TODO doesn't work

C = sch.get_block("C", func_name="bmm_relu_almost")
i0, i1, i2_0, i2_1 = sch.get_loops(C)
sch.parallel(loop=i0) # TODO doesn't work


