import torch
import numpy as np
import IPython
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T

I, J, K = 3, 3, 3
in_shape = (I,J)

@tvm.script.ir_module
class AlmostParal:
    @T.prim_func
    def mm(A: T.Buffer(in_shape, "float32"), B: T.Buffer(in_shape, "float32"), C: T.Buffer(in_shape, "float32")):
        T.func_attr({"global_symbol": "mm", "tir.noalias": True})
        for i1, i2 in T.grid(J, K):
            for ax_0 in T.grid(K):
                with T.block("C"):
                    i, j = T.axis.remap("SS", [i1, i2])
                    k = T.axis.reduce(K, ax_0) # TODO this line is the issue!
                    C[i, j] = C[i, j] + A[i, k] * B[k, j]

a = np.random.rand(*in_shape).astype("float32")
b = np.random.rand(*in_shape).astype("float32")
c = np.random.rand(*in_shape).astype("float32")
expected = c.copy()
for i in range(I):
    for j in range(J):
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
# Y_init = sch.get_block("Y_init", func_name="bmm_relu")
# Y_update = sch.get_block("Y_update", func_name="bmm_relu")
C = sch.get_block("C", func_name="bmm_relu")
# i0, i1, i2_0, i2_1_init = sch.get_loops(Y_init)
# i0, i1, i2_0, i2_1, ax_0, ax_1 = sch.get_loops(Y_update)
i0, i1, i2, ax_0 = sch.get_loops(C)
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