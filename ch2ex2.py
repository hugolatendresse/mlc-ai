import torch
import numpy as np
import IPython
import tvm
from tvm.ir.module import IRModule
from tvm.script import tir as T

# H and W are the input image height and width
# K is the 1D size of the convolution weight
# OUT_H is the height of the output image, assuming a stride of 1 and padding of 0
N, CI, H, W, CO, K = 1, 1, 8, 8, 2, 3
OUT_H, OUT_W = H - K + 1, W - K + 1
data = np.arange(N*CI*H*W).reshape(N, CI, H, W)
weight = np.arange(CO*CI*K*K).reshape(CO, CI, K, K)

data_torch = torch.Tensor(data)
weight_torch = torch.Tensor(weight)
conv_torch = torch.nn.functional.conv2d(data_torch, weight_torch)
conv_torch = conv_torch.numpy().astype(np.int64)
conv_torch


@tvm.script.ir_module
class MyConv:
    @T.prim_func
    def conv(DATA: T.buffer(shape=(N, CI, H, W), dtype="int64"),
             WEIGHT: T.buffer(shape=(CO, CI, K, K), dtype="int64"),
             CONV: T.buffer(shape=(N, CO, OUT_H, OUT_W), dtype="int64")):
        T.func_attr({"global_symbol": "conv", "tir.noalias": True})
        for b, k, i, j, di, dj, q in T.grid(N, CO, OUT_H, OUT_W, K, K, CI):
            with T.block("CONV"):
                vb = T.axis.spatial(N, b)
                vk = T.axis.spatial(CO, k)
                vi = T.axis.spatial(OUT_H, i)
                vj = T.axis.spatial(OUT_W, j)
                vdi = T.axis.reduce(K, di)
                vdj = T.axis.reduce(K, dj)
                vq = T.axis.reduce(CI, q)
                with T.init():
                    CONV[vb, vk, vi, vj] = T.int64(0)
                CONV[vb, vk, vi, vj] = CONV[vb, vk, vi, vj] + DATA[vb, vq, vi + vdi, vj + vdj] * WEIGHT[vk, vq, vdi, vdj]

rt_lib = tvm.build(MyConv, target="llvm")
data_tvm = tvm.nd.array(data)
weight_tvm = tvm.nd.array(weight)
conv_tvm = tvm.nd.array(np.empty((N, CO, OUT_H, OUT_W), dtype=np.int64))
rt_lib["conv"](data_tvm, weight_tvm, conv_tvm)
np.testing.assert_allclose(conv_tvm.numpy(), conv_torch, rtol=1e-5)