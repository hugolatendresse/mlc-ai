import torch
import numpy as np
import IPython
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T

N, I, J, K = 16, 128, 128, 128
in_shape = (N,I,J)

@tvm.script.ir_module
class AlmostParal:
    @T.prim_func
    def bmm_relu(A: T.Buffer((16, 128, 128), "float32"), B: T.Buffer((16, 128, 128), "float32"), C: T.Buffer((16, 128, 128), "float32")):
        T.func_attr({"global_symbol": "bmm_relu", "tir.noalias": True})
        # with T.block("root"):
        Y = T.alloc_buffer((16, 128, 128))
        for i0 in T.grid(16):
            for i1, i2_0 in T.grid(128, 16):
                # for i2_1_init in T.vectorized(8):
                #     with T.block("Y_init"):
                #         n, i = T.axis.remap("SS", [i0, i1])
                #         j = T.axis.spatial(128, i2_0 * 8 + i2_1_init)
                #         # T.reads()
                #         # T.writes(Y[n, i, j])
                #         Y[n, i, j] = T.float32(0.0)
                for i2_1, ax_0, ax_1 in T.grid(8, 32, 4):
                    with T.block("Y_update"):
                        # n, i = T.axis.remap("SS", [i0, i1])
                        # j = T.axis.spatial(128, i2_0 * 8 + i2_1)
                        k = T.axis.reduce(128, ax_0 * 4 + ax_1) # TODO this line is the issue!
                #         # T.reads(Y[n, i, j], A[n, i, k], B[n, k, j])
                #         # T.writes(Y[n, i, j])
                #         Y[n, i, j] = Y[n, i, j] + A[n, i, k] * B[n, k, j]
                # for ax0 in range(8):
                #     with T.block("C"):
                #         n, i = T.axis.remap("SS", [i0, i1])
                #         j = T.axis.spatial(128, i2_0 * 8 + ax0)
                #         # T.reads(Y[n, i, j])
                #         # T.writes(C[n, i, j])
                #         C[n, i, j] = T.max(Y[n, i, j], T.float32(0.0))


a = np.random.rand(*in_shape).astype("float32")
n = np.random.rand(*in_shape).astype("float32")
expected = a @ n

rt_lib1 = tvm.build(AlmostParal, target="llvm")
a_tvm1 = tvm.nd.array(a)
b_tvm1 = tvm.nd.array(n)
c_tvm1 = tvm.nd.array(np.random.rand(16, 128, 128).astype("float32"))
rt_lib1["bmm_relu"](a_tvm1, b_tvm1, c_tvm1)
almost_paral = c_tvm1.numpy()
# print(almost_paral)


sch = tvm.tir.Schedule(AlmostParal)
# Y_init = sch.get_block("Y_init", func_name="bmm_relu")
Y_update = sch.get_block("Y_update", func_name="bmm_relu")
# i0, i1, i2_0, i2_1_init = sch.get_loops(Y_init)
i0, i1, i2_0, i2_1, ax_0, ax_1 = sch.get_loops(Y_update)
sch.parallel(loop=i0) # TODO comprendre pourquoi Almost Paral ne peut pas etre converti!!!
sch.mod.show()

# TODO
# sch.unroll(...)
# ...
#
# IPython.display.Code(sch.mod.script(), language="python")
#
# tvm.ir.assert_structural_equal(sch.mod, TargetModule)
# print("Pass")
#
# before_rt_lib = tvm.build(MyBmmRelu, target="llvm")
# after_rt_lib = tvm.build(sch.mod, target="llvm")
# a_tvm = tvm.nd.array(np.random.rand(16, 128, 128).astype("float32"))
# b_tvm = tvm.nd.array(np.random.rand(16, 128, 128).astype("float32"))
# c_tvm = tvm.nd.array(np.random.rand(16, 128, 128).astype("float32"))
# after_rt_lib["bmm_relu"](a_tvm, b_tvm, c_tvm)
# before_timer = before_rt_lib.time_evaluator("bmm_relu", tvm.cpu())
# print("Before transformation:")
# print(before_timer(a_tvm, b_tvm, c_tvm))
#
# f_timer = after_rt_lib.time_evaluator("bmm_relu", tvm.cpu())
# print("After transformation:")
# print(f_timer(a_tvm, b_tvm, c_tvm))