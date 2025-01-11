import torch
import numpy as np
import IPython
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T

I, J, K = 8, 8, 8
in_shape = (I,J)


@tvm.script.ir_module
class MatMul:
    @T.prim_func
    def mm(A: T.Buffer(in_shape, "float32"), B: T.Buffer(in_shape, "float32"), C: T.Buffer(in_shape, "float32")):
        T.func_attr({"global_symbol": "mm", "tir.noalias": T.bool(True)})
        for i0, i1, ax_0 in T.grid(I, J, K):
            with T.block("C"):
                i, j, k = T.axis.remap("SSR", [i0, i1, ax_0])
                with T.init():
                    C[i, j] = T.float32(0)
                C[i, j] = C[i, j] + A[i, k] * B[k, j]


a = np.random.rand(*in_shape).astype("float32")
b = np.random.rand(*in_shape).astype("float32")
c = np.random.rand(*in_shape).astype("float32")
expected = np.random.rand(*in_shape).astype("float32")

for i in range(I):
    for j in range(J):
        expected[i,j] = 0.0
        for k in range(K):
            expected[i,j] = expected[i,j] + a[i,k] * b[k, j]

rt_lib1 = tvm.build(MatMul, target="llvm")
a_tvm1 = tvm.nd.array(a)
b_tvm1 = tvm.nd.array(b)
c_tvm1 = tvm.nd.array(c)
rt_lib1["mm"](a_tvm1, b_tvm1, c_tvm1)
almost_paral = c_tvm1.numpy()
# print(almost_paral)

np.testing.assert_allclose(actual=almost_paral, desired=expected)

sch = tvm.tir.Schedule(MatMul)


C = sch.get_block("C", func_name="mm")
i0, i1, ax0 = sch.get_loops(C)
sch.parallel(loop=i0) # able to parallelize with only one block!
IPython.display.Code(sch.mod.script(), language="python")



# Final check for correctness
rt_lib2 = tvm.build(sch.mod, target="llvm")
a_tvm2 = tvm.nd.array(a)
b_tvm2 = tvm.nd.array(b)
c_tvm2 = tvm.nd.array(c)
rt_lib2["mm"](a_tvm2, b_tvm2, c_tvm2)
done2 = c_tvm2.numpy()
np.testing.assert_allclose(actual=done2, desired=expected)




