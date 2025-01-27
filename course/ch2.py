import IPython
import numpy as np
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T

# high-level numpy program for linear + relu
dtype = "float32"
a_np = np.random.rand(128, 128).astype(dtype)
b_np = np.random.rand(128, 128).astype(dtype)
c_mm_relu = np.maximum(a_np @ b_np, 0)

# low-level numpy program for linear + relu
def lnumpy_mm_relu(A: np.ndarray, B: np.ndarray, C: np.ndarray):
    Y = np.empty((128, 128), dtype="float32")
    for i in range(128):
        for j in range(128):
            for k in range(128):
                if k == 0:
                    Y[i, j] = 0
                Y[i, j] = Y[i, j] + A[i, k] * B[k, j]
    for i in range(128):
        for j in range(128):
            C[i, j] = max(Y[i, j], 0)

c_np = np.empty((128, 128), dtype=dtype)
lnumpy_mm_relu(a_np, b_np, c_np)
np.testing.assert_allclose(c_mm_relu, c_np, rtol=1e-5)

# TensorIR representation of mm_relu
@tvm.script.ir_module # This decorator indicates an IRModule, which contains tensor funcs
class MyModule:
    @T.prim_func # decorator for a tensor function. There can be more than one in a given IRmodule
    def mm_relu(A: T.Buffer((128, 128), "float32"),
                B: T.Buffer((128, 128), "float32"),
                C: T.Buffer((128, 128), "float32")):
        # global_symbol is the name of the function
        # tir.noalis means that buffer memory areas do not overlap
        T.func_attr({"global_symbol": "mm_relu", "tir.noalias": True})
        Y = T.alloc_buffer((128, 128), dtype="float32")
        for iY, jY, kY in T.grid(128, 128, 128):
            with T.block("Y"):
                # Note: the three lines below are equivalent to just this one line
                # vi, vj, vk = T.axis.remap("SSR", [i,j,k]) # SSR means spatial spatial reduce
                vi = T.axis.spatial(128, iY)
                vj = T.axis.spatial(128, jY)
                vk = T.axis.reduce(128, kY)
                with T.init():
                    Y[vi, vj] = T.float32(0)
                Y[vi, vj] = Y[vi, vj] + A[vi, vk] * B[vk, vj]
        for iC, jC in T.grid(128, 128):
            with T.block("C"):
                # Two lines below are equivalent to T.axis.remap("SS", [i,j])
                vi = T.axis.spatial(128, iC)
                vj = T.axis.spatial(128, jC)
                C[vi, vj] = T.max(Y[vi, vj], T.float32(0))

sch = tvm.tir.Schedule(MyModule)
block_Y = sch.get_block("Y", func_name="mm_relu")
iY, jY, kY = sch.get_loops(block_Y)
jY_0, jY_1 = sch.split(jY, factors=[None, 4])
sch.reorder(jY_0, kY, jY_1)
block_C = sch.get_block("C", "mm_relu")
IPython.display.Code(sch.mod.script(), language="python")
sch.reverse_compute_at(block_C, jY_0)
IPython.display.Code(sch.mod.script(), language="python")

# IPython.display.Code(sch.mod.script(), language="python")
sch.decompose_reduction(block_Y, kY)
# IPython.display.Code(sch.mod.script(), language='python')

rt_lib = tvm.build(MyModule, target='llvm')
a_nd = tvm.nd.array(a_np)
b_nd = tvm.nd.array(b_np)
c_nd = tvm.nd.empty((128,128), dtype="float32")
type(c_nd)

func_mm_relu = rt_lib["mm_relu"]
func_mm_relu(a_nd, b_nd, c_nd)
np.testing.assert_allclose(c_mm_relu, c_nd.numpy(), rtol=1e-5)

rt_lib_after = tvm.build(sch.mod, target="llvm")
rt_lib_after["mm_relu"](a_nd, b_nd, c_nd)
np.testing.assert_allclose(c_mm_relu, c_nd.numpy(), rtol=1e-5)

f_timer_before = rt_lib.time_evaluator("mm_relu", tvm.cpu())
print("Time cost of MyModule %g sec" % f_timer_before(a_nd, b_nd, c_nd).mean)
f_timer_after = rt_lib_after.time_evaluator("mm_relu", tvm.cpu())
print("Time cost of transformed sch.mod %g sec" % f_timer_after(a_nd, b_nd, c_nd).mean)

f_timer_before(a_nd, b_nd, c_nd).mean / f_timer_after(a_nd, b_nd, c_nd).mean

def transform(mod, jfactor):
    sch = tvm.tir.Schedule(mod)
    block_Y = sch.get_block("Y", func_name="mm_relu")
    i, j, k = sch.get_loops(block_Y)
    j0, j1 = sch.split(j, factors=[None, jfactor])
    sch.reorder(j0, k, j1)
    block_C = sch.get_block("C", "mm_relu")
    sch.reverse_compute_at(block_C, j0)
    return sch.mod

mod_transformed = transform(MyModule, jfactor=8)

rt_lib_transformed = tvm.build(mod_transformed, "llvm")
f_timer_transformed = rt_lib_transformed.time_evaluator("mm_relu", tvm.cpu())
print("Time cost of transformed mod_transformed %g sec" % f_timer_transformed(a_nd, b_nd, c_nd).mean)
# display the code below
IPython.display.Code(mod_transformed.script(), language="python")




# Using Tensor Expressions
from tvm import te
A = te.placeholder((128,128), "float32", name='A')
B = te.placeholder((128, 128), "float32", name='B')
kY = te.reduce_axis((0, 128), 'k')
Y = te.compute((128, 128), lambda i, j: te.sum(A[i,kY] * B[kY, j], axis=kY), name='Y')
C = te.compute((128,128), lambda i, j: te.max(Y[i,j], 0), name="C")

te_func = te.create_prim_func([A,B,C]).with_attr({"global_symbol": "mm_relu"})
MyModuleFromTE = tvm.IRModule({"mm_relu": te_func})
IPython.display.Code(MyModuleFromTE.script(), language='python')