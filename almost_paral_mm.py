import torch
import numpy as np
import IPython
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T

I, J, K = 8, 8, 8
in_shape = (I,J)


@tvm.script.ir_module
class AlmostParal:
    @T.prim_func
    def mm(A: T.Buffer((8, 8), "float32"), B: T.Buffer((8, 8), "float32"), C: T.Buffer((8, 8), "float32")):
        T.func_attr({"tir.noalias": T.bool(True)})
        for i0_i1 in T.parallel(64):  # Fused i0 and i1 for parallelism
            i0 = i0_i1 // 8
            i1 = i0_i1 % 8
            with T.block("C_init"):
                i, j = T.axis.remap("SS", [i0, i1])
                T.reads()
                T.writes(C[i, j])
                C[i, j] = 0.0
            for ax_0 in range(8):
                with T.block("C"):
                    i, j, k = T.axis.remap("SSR", [i0, i1, ax_0])
                    T.reads(C[i, j], A[i, k], B[k, j])
                    T.writes(C[i, j])
                    C[i, j] = C[i, j] + A[i, k] * B[k, j]


a = np.random.rand(*in_shape).astype("float32")
b = np.random.rand(*in_shape).astype("float32")
c = np.random.rand(*in_shape).astype("float32")
expected = c.copy()
for i in range(I):
    for j in range(J):
        expected[i,j] = 0.0
        for k in range(K):
            expected[i,j] = expected[i,j] + a[i,k] * b[k, j]

rt_lib1 = tvm.build(AlmostParal, target="llvm")
a_tvm1 = tvm.nd.array(a)
b_tvm1 = tvm.nd.array(b)
c_tvm1 = tvm.nd.array(c)
rt_lib1["mm"](a_tvm1, b_tvm1, c_tvm1)
almost_paral = c_tvm1.numpy()
# print(almost_paral)

np.testing.assert_allclose(actual=almost_paral, desired=expected)

sch = tvm.tir.Schedule(AlmostParal)
C_init = sch.get_block("C_init", func_name="mm")
C = sch.get_block("C", func_name="mm")
toute = sch.get_block("toute", func_name="mm")
i0_i1 = sch.get_loops(C_init)
i0_i1, ax0 = sch.get_loops(C)

# TODO comprendre pourquoi parallel doesn't work!!!
sch.parallel(i0)
sch.parallel(loop=i0)
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