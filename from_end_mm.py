import torch
import numpy as np
import IPython
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T

I, J, K = 16, 128, 128
in_shape = (I,J, K)

@tvm.script.ir_module
class TargetModule:
    @T.prim_func
    def mm(A: T.Buffer((16, 128, 128), "float32"), B: T.Buffer((16, 128, 128), "float32"), C: T.Buffer((16, 128, 128), "float32")) -> None:
        T.func_attr({"global_symbol": "mm", "tir.noalias": True})
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
b = np.random.rand(*in_shape).astype("float32")
c = np.random.rand(*in_shape).astype("float32")
expected = c.copy()

rt_lib1 = tvm.build(TargetModule, target="llvm")
a_tvm1 = tvm.nd.array(a)
b_tvm1 = tvm.nd.array(b)
c_tvm1 = tvm.nd.array(c)
rt_lib1["mm"](a_tvm1, b_tvm1, c_tvm1)
almost_paral = c_tvm1.numpy()

sch = tvm.tir.Schedule(TargetModule)
C = sch.get_block("C", func_name="mm")
i0, i1, i2_0, i2_1 = sch.get_loops(C)
sch.parallel(loop=i1) # TODO comprendre pourquoi Almost Paral ne peut pas etre converti!!!
sch.mod.show()
