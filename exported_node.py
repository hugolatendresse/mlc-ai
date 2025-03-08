# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=invalid-name, inconsistent-return-statements, unidiomatic-typecheck
# pylint: disable=import-outside-toplevel
"""Base class for PyTorch FX Graph importer."""
import abc
from typing import Callable, Dict, Optional, Tuple, Union

from tvm import relax

def mod(x1, x2):
    """Modulo with numpy-style broadcasting.

    Parameters
    ----------
    x1 : Expr
        The first input tensor.
    x2 : Expr
        The second input tensor.
    """
    return _ffi_api.mod(x1, x2)  # type: ignore



class BaseFXGraphImporter(metaclass=abc.ABCMeta):
    """Base class for FX Graph Importer."""

    import torch  # type: ignore
    from torch import fx

    def __init__(self) -> None:
        import torch  # type: ignore
        from torch import fx

        self.env: Dict[fx.Node, relax.Expr] = {}
        self.params: Dict[torch.Tensor, relax.Expr] = {}
        self.block_builder: relax.BlockBuilder = None
        self.convert_map: Dict[
            Union[torch.nn.Module, str], Callable[[fx.Node], relax.Var]
        ] = self.create_convert_map()

    ########## Utilities ##########

    @staticmethod
    def _convert_data_type(input_type: Union[str, torch.dtype], env: Optional[Dict] = None):
        """converts the PyTorch scalar type input_type to a TVM dtype."""
        import torch  # type: ignore

        if env is not None and input_type in env:
            input_type = env[input_type]

        input_type = input_type.lower() if isinstance(input_type, str) else input_type
        if input_type in ["float", "float32", "torch.float32", torch.float32]:
            return "float32"
        elif input_type in ["float16", "torch.float16", torch.float16]:
            return "float16"
        elif input_type in ["int64", "torch.int64", torch.int64]:
            return "int64"
        elif input_type in ["int32", "torch.int32", torch.int32]:
            return "int32"
        elif input_type in ["bool", "torch.bool", torch.bool]:
            return "bool"
        else:
            raise NotImplementedError("input_type {} is not handled yet".format(input_type))

    @staticmethod
    def _convert_torch_tensor_to_relax(tensor: torch.Tensor) -> relax.Var:
        tensor = tensor.detach().cpu()
        dtype = BaseFXGraphImporter._convert_data_type(str(tensor.data.dtype))
        return relax.const(tensor.data.numpy(), dtype)

    @staticmethod
    def shape_of(tensor):
        """Get the shape of a tensor."""
        import torch  # type: ignore

        if isinstance(tensor, relax.Expr):
            if not isinstance(tensor.struct_info, relax.TensorStructInfo):
                raise TypeError("The input Expr of shape_of should be a Tensor")
            return tensor.struct_info.shape
        elif isinstance(tensor, torch.Tensor):
            return tensor.shape
        raise ValueError("Unsupported type: {}".format(type(tensor)))

    def retrieve_args(self, node: fx.Node):
        return self._retrieve_args(node.args)

    def _retrieve_args(self, node):
        from torch import fx

        if isinstance(node, fx.Node):
            return self.env[node]
        elif isinstance(node, tuple):
            return tuple(self._retrieve_args(x) for x in node)
        elif isinstance(node, list):
            return [self._retrieve_args(x) for x in node]
        elif isinstance(node, dict):
            return {self._retrieve_args(k): self._retrieve_args(v) for k, v in node.items()}
        else:
            return node

    ########## Unary Ops ##########

    def _unary_op(self, op: Callable) -> Callable:
        from torch import fx

        def convert(node: fx.Node) -> relax.Var:
            try:
                return self.block_builder.emit(op(self.env[node.args[0]]))
            except:
                print("THIS NODE FAILED!!!!")
                print(node)
                print(node.args)
                print(node.kwargs)
                print(id(node))
                print(dir(node))
                print("node name:", node.name)
                print("node stack trace", node.stack_trace)
                print("node type", node.type)
                print("DONE PRINTING NODE FAILED")
                raise Exception("Node failed!!!!")

        return convert

    def _celu(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        alpha = node.args[1] if len(node.args) > 1 else node.kwargs.get("alpha", 1.0)
        dtype = x.struct_info.dtype

        if isinstance(alpha, (int, float)):
            alpha = relax.const(alpha, dtype)
        else:
            if not isinstance(alpha, relax.Var):
                alpha = self.block_builder.emit(relax.const(alpha, dtype))

        zero = relax.const(0, dtype)
        # alpha * min(0, exp(x / alpha) - 1) + max(0, x)
        return self.block_builder.emit(
            relax.op.add(
                relax.op.multiply(
                    alpha,
                    relax.op.minimum(
                        zero,
                        relax.op.subtract(
                            relax.op.divide(relax.op.exp(x), alpha), relax.const(1, dtype)
                        ),
                    ),
                ),
                relax.op.nn.relu(x),
            )
        )

    def _clamp(self, node: fx.Node) -> relax.Expr:
        args = self.retrieve_args(node)
        a_min = args[1] if len(args) > 1 else node.kwargs["min"]
        a_max = args[2] if len(args) > 2 else node.kwargs["max"]
        if not isinstance(a_min, (int, float)):
            raise ValueError(
                f"TVM only supports constant min value for torch.clamp/clip, "
                f"but got {a_min} with type {type(a_min)}"
            )
        if not isinstance(a_max, (int, float)):
            raise ValueError(
                f"TVM only supports constant max value for torch.clamp/clip, "
                f"but got {a_max} with type {type(a_max)}"
            )
        return self.block_builder.emit(relax.op.clip(args[0], a_min, a_max))

    def _clamp_min(self, node: fx.Node) -> relax.Expr:
        args = self.retrieve_args(node)
        a_min = args[1] if len(args) > 1 else node.kwargs["min"]
        import math
        a_max = math.inf
        if not isinstance(a_min, (int, float)):
            raise ValueError(
                f"TVM only supports constant min value for torch.clamp/clip, "
                f"but got {a_min} with type {type(a_min)}"
            )
        if not isinstance(a_max, (int, float)):
            raise ValueError(
                f"TVM only supports constant max value for torch.clamp/clip, "
                f"but got {a_max} with type {type(a_max)}"
            )
        return self.block_builder.emit(relax.op.clip(args[0], a_min, a_max))

    def _elu(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        alpha = node.args[1] if len(node.args) > 1 else node.kwargs.get("alpha", 1.0)
        dtype = x.struct_info.dtype

        if isinstance(alpha, (int, float)):
            alpha = relax.const(-alpha, dtype)
        else:
            if not isinstance(alpha, relax.Var):
                alpha = self.block_builder.emit(relax.const(-alpha, dtype))

        # alpha * ReLU(1 − exp(x)) + ReLU(x)
        return self.block_builder.emit(
            relax.op.add(
                relax.op.multiply(
                    alpha,
                    relax.op.nn.relu(relax.op.subtract(relax.const(1, dtype), relax.op.exp(x))),
                ),
                relax.op.nn.relu(x),
            )
        )

    def _gelu(self, node: fx.Node) -> relax.Expr:
        approximate = node.kwargs.get("approximate", "none")
        if approximate == "none":
            return self.block_builder.emit(relax.op.nn.gelu(self.env[node.args[0]]))
        elif approximate == "tanh":
            return self.block_builder.emit(relax.op.nn.gelu_tanh(self.env[node.args[0]]))
        else:
            raise KeyError("Unregonized approximate algorithm for gelu: {}.".format(approximate))

    def _hardsigmoid(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        dtype = x.struct_info.dtype
        x0 = relax.op.add(x, relax.const(3, dtype))
        x1 = relax.op.clip(x0, 0, 6)
        return self.block_builder.emit(relax.op.divide(x1, relax.const(6, dtype)))

    def _hardswish(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        dtype = x.struct_info.dtype
        x0 = relax.op.add(x, relax.const(3, dtype))
        x1 = relax.op.clip(x0, 0, 6)
        x2 = relax.op.divide(x1, relax.const(6, dtype))
        return self.block_builder.emit(relax.op.multiply(x, x2))

    def _hardtanh(self, node: fx.Node) -> relax.Expr:
        args = self.retrieve_args(node)
        x = args[0]
        min_val = node.kwargs.get("min_val", -1.0)
        max_val = node.kwargs.get("max_val", 1.0)
        return self.block_builder.emit(relax.op.clip(x, min_val, max_val))

    def _leakyrelu(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        alpha = node.args[1] if len(node.args) > 1 else node.kwargs.get("negative_slope", 0.01)
        return self.block_builder.emit(relax.op.nn.leakyrelu(x, alpha))

    def _log_softmax(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", -1)
        return self.block_builder.emit(relax.op.nn.log_softmax(x, dim))

    def _round(self, node: fx.Node) -> relax.Expr:
        if node.kwargs.get("decimals", 0) != 0:
            raise ValueError("specifying decimals for round is not supported yet")
        arg = self.env[node.args[0]]
        return self.block_builder.emit(relax.op.round(arg))

    def _softmax(self, node: fx.Node) -> relax.Var:
        """
        Enhanced softmax implementation that handles large tensors with non-last dimensions.
        For large tensors with non-last dimension softmax, we transpose to move the softmax
        dimension to the end, apply softmax, then transpose back to the original shape.
        """
        print("ENTERING _softmax() in base_fx_graph_importer.py")
        print("the kwargs type is", type(node.kwargs))
        print("the raw kwargs are: ", node.kwargs)
        print("the kwargs are: ", list(node.kwargs.keys()))
        x = self.env[node.args[0]]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", -1)

        # Get information about the input tensor
        input_shape = x.struct_info.shape
        input_ndim = len(input_shape)

        # Normalize negative dim
        if dim < 0:
            dim = input_ndim + dim

        # Check if this is a non-last dimension with large size (>4096)
        is_large_non_last_dim = False
        large_size_threshold = 4096

        if dim != input_ndim - 1:  # Not the last dimension
            try:
                # Check if any dimension is large
                for i, size in enumerate(input_shape):
                    if hasattr(size, 'value') and size.value > large_size_threshold:
                        is_large_non_last_dim = True
                        break
            except:
                # If we can't determine the size, play it safe
                pass

        print("_softmax about to call relax.op.nn.softmax")

        if is_large_non_last_dim:
            # Special handling for large tensors with non-last dimension softmax

            # 1. Get the dimension ordering for transpose
            dims = list(range(input_ndim))
            # Move the softmax dimension to the end
            dims.append(dims.pop(dim))

            # 2. Transpose to move softmax dim to the end
            x_transposed = self.block_builder.emit(relax.op.permute_dims(x, dims))

            # 3. Apply softmax on the last dimension
            softmax_result = self.block_builder.emit(relax.op.nn.softmax(x_transposed, -1))

            # 4. Transpose back to original shape
            # Calculate the inverse permutation
            inv_dims = [-1] * len(dims)
            for i, d in enumerate(dims):
                inv_dims[d] = i

            # 5. Apply the inverse permutation
            result = self.block_builder.emit(relax.op.permute_dims(softmax_result, inv_dims))
        else:
            # Regular softmax for last dimension or small tensors
            result = self.block_builder.emit(relax.op.nn.softmax(x, dim))

        return result

    def _selu(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        alpha = node.args[1] if len(node.args) > 1 else node.kwargs.get("alpha", 1.6732631921768188)
        gamma = node.args[2] if len(node.args) > 2 else node.kwargs.get("gamma", 1.0507009873554805)
        dtype = x.struct_info.dtype

        if isinstance(alpha, (int, float)):
            alpha = relax.const(alpha, dtype)
        else:
            if not isinstance(alpha, relax.Var):
                alpha = self.block_builder.emit(relax.const(alpha, dtype))

        if isinstance(gamma, (int, float)):
            gamma = relax.const(gamma, dtype)
        else:
            if not isinstance(gamma, relax.Var):
                gamma = self.block_builder.emit(relax.const(gamma, dtype))

        # gamma * (ReLU(x) + alpha * (exp(x) - 1))
        return self.block_builder.emit(
            relax.op.multiply(
                gamma,
                relax.op.add(
                    relax.op.nn.relu(x),
                    relax.op.multiply(
                        alpha, relax.op.subtract(relax.op.exp(x), relax.const(1, dtype))
                    ),
                ),
            )
        )

    def _tril_triu(self, op: Callable) -> Callable:
        from torch import fx

        def convert(node: fx.Node) -> relax.Var:
            x = self.env[node.args[0]]
            k = node.args[1] if len(node.args) > 1 else node.kwargs.get("diagonal", 0)
            assert isinstance(k, int)
            return self.block_builder.emit(op(x, k))

        return convert

    ########## Binary Ops ##########

    def _binary_op(self, relax_op: Callable, intrinsic_op: Callable) -> Callable:
        from torch import fx

        def convert(node: fx.Node) -> relax.Var:
            def promote_binary_op_args(lhs, rhs):
                if isinstance(lhs, relax.Expr) and isinstance(rhs, relax.Expr):
                    return lhs, rhs
                elif isinstance(lhs, relax.Expr):
                    assert isinstance(lhs.struct_info, relax.TensorStructInfo)
                    return lhs, relax.const(rhs, lhs.struct_info.dtype)
                elif isinstance(rhs, relax.Expr):
                    assert isinstance(rhs.struct_info, relax.TensorStructInfo)
                    return relax.const(lhs, rhs.struct_info.dtype), rhs
                else:
                    assert False

            def call_binary_op(op, lhs, rhs):
                lhs, rhs = promote_binary_op_args(lhs, rhs)
                return self.block_builder.emit(op(lhs, rhs))

            lhs, rhs = self.retrieve_args(node)
            if isinstance(lhs, relax.Var) or isinstance(rhs, relax.Var):
                return call_binary_op(relax_op, lhs, rhs)
            elif isinstance(lhs, relax.expr.Constant):
                return call_binary_op(relax_op, lhs, relax.const(rhs, dtype=lhs.struct_info.dtype))
            elif isinstance(rhs, relax.expr.Constant):
                return call_binary_op(relax_op, relax.const(lhs, dtype=rhs.struct_info.dtype), rhs)
            return intrinsic_op(lhs, rhs)

        return convert

    ########## Linear Algebra ##########

    def _linalg_vector_norm(self, node: fx.Node) -> relax.Var:

        args = self.retrieve_args(node)

        # The PyTorch signature is typically something like
        # torch.linalg.vector_norm(input, ord=2, dim=None, keepdim=False, dtype=None)
        # Adjust as needed depending on how you retrieve arguments.
        data = args[0]
        # Default ord=2 if not supplied
        ord_val = args[1] if len(args) > 1 else 2.0
        dim = args[2] if len(args) > 2 else None
        keepdim = args[3] if len(args) > 3 else False

        # If ord_val is a Python float/int, wrap it in a Relax const
        # so that it matches data's dtype.  If ord_val is already an Expr, you can skip this.
        dtype = data.struct_info.dtype
        ord_expr = (
            ord_val
            if isinstance(ord_val, relax.Expr)
            else relax.const(float(ord_val), dtype)
        )
        # Reciprocal
        reci_expr = (
            relax.op.divide(
                relax.const(1.0, dtype),
                ord_expr
            )
            if isinstance(ord_val, relax.Expr)
            else relax.const(1.0 / float(ord_val), dtype)
        )

        # abs(data)
        abs_data = self.block_builder.emit(relax.op.abs(data))
        # abs_data^ord
        abs_data_pow = self.block_builder.emit(relax.op.power(abs_data, ord_expr))
        # sum over dim
        reduced = self.block_builder.emit(relax.op.sum(abs_data_pow, dim, keepdims=keepdim))
        # (sum(...))^(1/ord)
        norm_val = self.block_builder.emit(relax.op.power(reduced, reci_expr))

        return norm_val

    ########## Neural Network ##########

    def _adaptive_avg_pool2d(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        output_size = node.args[1]
        return self.block_builder.emit(
            relax.op.nn.adaptive_avg_pool2d(x, output_size, layout="NCHW")
        )

    def _addmm(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        y = self.env[node.args[1]]
        z = self.env[node.args[2]]
        alpha = node.kwargs.get("alpha", 1)
        beta = node.kwargs.get("beta", 1)

        res = None
        if alpha != 0:
            res = self.block_builder.emit(relax.op.linear_algebra.matmul(y, z, out_dtype="float32"))
            if alpha != 1:
                dtype = res.struct_info.dtype
                res = self.block_builder.emit(relax.op.multiply(res, relax.const(alpha, dtype)))
        if beta != 0:
            dtype = x.struct_info.dtype
            if beta != 1:
                bias = self.block_builder.emit(relax.op.multiply(x, relax.const(beta, dtype)))
            else:
                bias = x
            res = bias if res is None else self.block_builder.emit(relax.op.add(bias, res))
        return res

    def _avg_pool2d_impl(
            self,
            x: relax.Expr,
            kernel_size: Union[int, Tuple[int, int]] = (1, 1),
            stride: Optional[Union[int, Tuple[int, int]]] = None,
            padding: Optional[int] = 0,
            ceil_mode: Optional[bool] = False,
    ) -> relax.Var:
        stride = kernel_size if stride is None or stride == [] else stride
        return self.block_builder.emit(
            relax.op.nn.avg_pool2d(
                x,
                pool_size=kernel_size,
                strides=stride,
                padding=padding,
                ceil_mode=ceil_mode,
                layout="NCHW",
            )
        )

    def _avg_pool2d(self, node: fx.Node) -> relax.Var:
        args, kwargs = node.normalized_arguments(node)
        x = self.env[args[0]]
        kernel_size = args[1] if len(args) > 1 else kwargs["kernel_size"]
        stride = args[2] if len(args) > 2 else kwargs.get("stride", None)
        padding = args[3] if len(args) > 3 else kwargs.get("padding", 0)
        ceil_mode = args[4] if len(args) > 4 else kwargs.get("ceil_mode", False)
        return self._avg_pool2d_impl(x, kernel_size, stride, padding, ceil_mode)

    def _baddbmm(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        batch1 = self.env[node.args[1]]
        batch2 = self.env[node.args[2]]
        alpha = node.kwargs.get("alpha", 1)
        beta = node.kwargs.get("beta", 1)

        res = None
        if alpha != 0:
            res = self.block_builder.emit(relax.op.matmul(batch1, batch2))
            if alpha != 1:
                dtype = res.struct_info.dtype
                res = self.block_builder.emit(relax.op.multiply(res, relax.const(alpha, dtype)))
        if beta != 0:
            dtype = x.struct_info.dtype
            if beta != 1:
                bias = self.block_builder.emit(relax.op.multiply(x, relax.const(beta, dtype)))
            else:
                bias = x
            res = bias if res is None else self.block_builder.emit(relax.op.add(res, bias))
        return res

    def _conv_transpose1d_impl(
            self,
            x: relax.Expr,
            weight: relax.Expr,
            bias: Optional[relax.Expr],
            strides: Optional[Tuple],
            padding: Optional[Tuple],
            dilation: Optional[Tuple],
            groups: Optional[Tuple],
    ) -> relax.Var:
        conv1d_transpose = self.block_builder.emit(
            relax.op.nn.conv1d_transpose(
                x,
                weight,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                data_layout="NCW",
                kernel_layout="OIW",
                out_dtype="float32",
            )
        )

        if bias is None:
            return conv1d_transpose

        assert len(self.shape_of(bias)) == 1
        bias = relax.op.reshape(bias, (1, -1, 1))
        return self.block_builder.emit(relax.op.add(conv1d_transpose, bias))

    def _conv_transpose1d(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None
        stride = args[3] if len(args) > 3 else 1
        padding = args[4] if len(args) > 4 else 0
        dilation = args[5] if len(args) > 5 else 1
        groups = args[6] if len(args) > 6 else 1
        return self._conv_transpose1d_impl(
            x,
            weight,
            bias=bias,
            strides=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    def _conv_transpose2d_impl(
            self,
            x: relax.Expr,
            weight: relax.Expr,
            bias: Optional[relax.Expr],
            strides: Optional[Tuple],
            padding: Optional[Tuple],
            dilation: Optional[Tuple],
            groups: Optional[Tuple],
    ) -> relax.Var:
        conv2d_transpose = self.block_builder.emit(
            relax.op.nn.conv2d_transpose(
                x,
                weight,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                data_layout="NCHW",
                kernel_layout="OIHW",
                out_dtype="float32",
            )
        )

        if bias is None:
            return conv2d_transpose

        assert len(self.shape_of(bias)) == 1
        bias = relax.op.reshape(bias, (1, -1, 1, 1))
        return self.block_builder.emit(relax.op.add(conv2d_transpose, bias))

    def _conv_transpose2d(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None
        stride = args[3] if len(args) > 3 else 1
        padding = args[4] if len(args) > 4 else 0
        dilation = args[5] if len(args) > 5 else 1
        groups = args[6] if len(args) > 6 else 1
        return self._conv_transpose2d_impl(
            x,
            weight,
            bias=bias,
            strides=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    def _conv1d_impl(
            self,
            x: relax.Expr,
            weight: relax.Expr,
            bias: Optional[relax.Expr],
            strides: Optional[Tuple],
            padding: Optional[Tuple],
            dilation: Optional[Tuple],
            groups: Optional[Tuple],
    ) -> relax.Var:
        conv1d = self.block_builder.emit(
            relax.op.nn.conv1d(
                x,
                weight,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                data_layout="NCW",
                kernel_layout="OIW",
                out_dtype="float32",
            )
        )

        if bias is None:
            return conv1d
        assert len(self.shape_of(bias)) == 1
        bias = relax.op.reshape(bias, (1, -1, 1))
        return self.block_builder.emit(relax.op.add(conv1d, bias))

    def _conv1d(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None
        stride = args[3] if len(args) > 3 else 1
        padding = args[4] if len(args) > 4 else 0
        dilation = args[5] if len(args) > 5 else 1
        groups = args[6] if len(args) > 6 else 1
        return self._conv1d_impl(
            x,
            weight,
            bias=bias,
            strides=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    def _conv2d_impl(
            self,
            x: relax.Expr,
            weight: relax.Expr,
            bias: Optional[relax.Expr],
            strides: Optional[Tuple],
            padding: Optional[Tuple],
            dilation: Optional[Tuple],
            groups: Optional[Tuple],
    ):
        conv2d = self.block_builder.emit(
            relax.op.nn.conv2d(
                x,
                weight,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                data_layout="NCHW",
                kernel_layout="OIHW",
                out_dtype="float32",
            )
        )

        if bias is None:
            return conv2d
        assert len(self.shape_of(bias)) == 1
        bias = relax.op.reshape(bias, (1, -1, 1, 1))
        return self.block_builder.emit(relax.op.add(conv2d, bias))

    def _conv2d(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None
        stride = args[3] if len(args) > 3 else 1
        padding = args[4] if len(args) > 4 else 0
        dilation = args[5] if len(args) > 5 else 1
        groups = args[6] if len(args) > 6 else 1
        return self._conv2d_impl(
            x,
            weight,
            bias=bias,
            strides=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    def _conv3d_impl(
            self,
            x: relax.Expr,
            weight: relax.Expr,
            bias: Optional[relax.Expr],
            strides: Optional[Tuple],
            padding: Optional[Tuple],
            dilation: Optional[Tuple],
            groups: Optional[Tuple],
    ):
        conv3d = self.block_builder.emit(
            relax.op.nn.conv3d(
                x,
                weight,
                strides=strides,
                padding=padding,
                dilation=dilation,
                groups=groups,
                data_layout="NCDHW",
                kernel_layout="OIDHW",
                out_dtype="float32",
            )
        )

        if bias is None:
            return conv3d
        assert len(self.shape_of(bias)) == 1
        bias = relax.op.reshape(bias, (1, -1, 1, 1, 1))
        return self.block_builder.emit(relax.op.add(conv3d, bias))

    def _conv3d(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None
        stride = args[3] if len(args) > 3 else 1
        padding = args[4] if len(args) > 4 else 0
        dilation = args[5] if len(args) > 5 else 1
        groups = args[6] if len(args) > 6 else 1
        return self._conv3d_impl(
            x,
            weight,
            bias=bias,
            strides=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
        )

    def _einsum(self, node: fx.Node) -> relax.Var:
        import torch  # type: ignore

        args = self.retrieve_args(node)
        operands = args[1] if isinstance(args[1], (torch.Size, tuple, list)) else args[1:]
        return self.block_builder.emit(relax.op.einsum(operands, args[0]))

    def _embedding_impl(
            self,
            x,
            weight,
    ) -> relax.Var:
        x = self.block_builder.emit(relax.op.astype(x, "int32"))

        ndim = x.struct_info.ndim
        if ndim == 1:
            return self.block_builder.emit(relax.op.take(weight, x, axis=0))
        else:
            x_shape = x.struct_info.shape.values
            emb_size = weight.struct_info.shape.values[-1]
            x = self.block_builder.emit(relax.op.reshape(x, shape=[-1]))
            embedding = self.block_builder.emit(relax.op.take(weight, x, axis=0))
            return self.block_builder.emit(relax.op.reshape(embedding, [*x_shape, emb_size]))

    def _layer_norm_impl(self, x, gamma, beta, eps, normalized_shape) -> relax.Var:
        from torch.fx.immutable_collections import immutable_list
        import numpy as np  # type: ignore

        if isinstance(normalized_shape, (immutable_list, tuple)):
            normalized_shape = tuple(normalized_shape)
        else:
            try:
                normalized_shape = self.env[normalized_shape]
            except TypeError:
                normalized_shape = tuple(normalized_shape)

        dim_num = len(normalized_shape)
        axes = list(range(-dim_num, 0))

        if gamma is None:
            shape_tuple = [int(s) for s in normalized_shape]
            gamma = relax.const(np.ones(shape_tuple), x.struct_info.dtype)
        if beta is None:
            shape_tuple = [int(s) for s in normalized_shape]
            beta = relax.const(np.zeros(shape_tuple), x.struct_info.dtype)

        return self.block_builder.emit(
            relax.op.nn.layer_norm(
                x,
                gamma,
                beta,
                axes=axes,
                epsilon=eps,
            )
        )

    def _layer_norm(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        normalized_shape = node.args[1]
        gamma = self.env[node.args[2]] if len(node.args) > 2 else None
        beta = self.env[node.args[3]] if len(node.args) > 3 else None
        eps = node.args[4] if len(node.args) > 4 else 1e-05
        return self._layer_norm_impl(x, gamma, beta, eps, normalized_shape)

    def _layer_norm_module(self, node: fx.Node) -> relax.Var:
        import torch  # type: ignore

        x = self.env[node.args[0]]
        module = self.named_modules[node.target]
        normalized_shape = module.normalized_shape
        if module.elementwise_affine:
            gamma = self.params[module.weight]
            beta = self.params[module.bias]
        else:
            gamma = relax.const(torch.ones_like(module.normalized_shape), x.struct_info.dtype)
            beta = relax.const(torch.zeros_like(module.normalized_shape), x.struct_info.dtype)
        eps = module.eps
        return self._layer_norm_impl(x, gamma, beta, eps, normalized_shape)

    def _linear(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        weight = args[1]
        bias = args[2] if len(args) > 2 else None
        return self.block_builder.emit(relax.op.linear(x, weight, bias, "float32"))

    def _max_pool2d_impl(
            self,
            x: relax.Expr,
            kernel_size: Union[int, Tuple[int, int]] = (1, 1),
            stride: Optional[Union[int, Tuple[int, int]]] = None,
            padding: Optional[int] = 0,
            dilation: Optional[int] = 1,
            ceil_mode: Optional[bool] = False,
    ) -> relax.Var:
        stride = kernel_size if stride is None else stride
        return self.block_builder.emit(
            relax.op.nn.max_pool2d(
                x,
                pool_size=kernel_size,
                strides=stride,
                padding=padding,
                dilation=dilation,
                ceil_mode=ceil_mode,
                layout="NCHW",
            )
        )

    def _max_pool2d(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        kernel_size = args[1]
        stride = args[2] if len(args) > 2 else None
        padding = args[3] if len(args) > 3 else 0
        dilation = args[4] if len(args) > 4 else 1
        ceil_mode = args[5] if len(args) > 5 else False

        return self._max_pool2d_impl(x, kernel_size, stride, padding, dilation, ceil_mode)

    def _scaled_dot_product_attention(self, node: fx.Node) -> relax.Var:
        transpose_S_H = lambda tensor: relax.op.permute_dims(tensor, [0, 2, 1, 3])
        query = transpose_S_H(self.env[node.args[0]])
        key = transpose_S_H(self.env[node.args[1]])
        value = transpose_S_H(self.env[node.args[2]])
        attn_mask = node.args[3] if len(node.args) > 3 else node.kwargs.get("attn_mask", None)
        dropout_p = node.args[4] if len(node.args) > 4 else node.kwargs.get("dropout_p", 0.0)
        assert dropout_p == 0.0, "Dropout is not supported"
        is_causal = node.args[5] if len(node.args) > 5 else node.kwargs.get("is_causal", False)
        causal_mask = "TopLeft" if is_causal else None

        if attn_mask is not None:
            attn_mask = self.env[attn_mask]
            msg = "Only a float mask is supported for the attn_mask input."
            assert "float" in attn_mask.struct_info.dtype, msg

        return self.block_builder.emit(
            transpose_S_H(
                relax.op.nn.attention(query, key, value, bias=attn_mask, causal_mask=causal_mask)
            )
        )

    def _unbind(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        assert isinstance(dim, int), "Expected 2nd argument of unbind as int"
        selections = self.shape_of(x)[dim].value
        n_section = list(range(1, selections + 1))
        ret, split = [], self.block_builder.emit(relax.op.split(x, n_section, dim))
        for i in range(selections):
            ret.append(self.block_builder.emit(relax.op.squeeze(split[i], axis=dim)))
        return self.block_builder.emit(relax.Tuple(ret))

    ########## Statistical ##########

    def _mean(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        dim = args[1] if len(node.args) > 1 else node.kwargs.get("dim", None)
        keepdim = args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
        return self.block_builder.emit(relax.op.mean(x, dim, keepdims=keepdim))

    def _sum(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        keepdim = node.kwargs["keepdim"] if "keepdim" in node.kwargs else False
        if len(args) == 1:
            return self.block_builder.emit(relax.op.sum(args[0], keepdims=keepdim))
        return self.block_builder.emit(relax.op.sum(args[0], args[1]))

    ########## Search ##########

    def _argmax_argmin(self, op: Callable) -> Callable:
        from torch import fx

        def convert(node: fx.Node):
            x = self.env[node.args[0]]
            dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", None)
            keepdim = node.args[2] if len(node.args) > 2 else node.kwargs.get("keepdim", False)
            return self.block_builder.emit(op(x, dim, keepdim))

        return convert

    ########## Manipulation ##########

    def _cat(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        axis = args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        return self.block_builder.emit(relax.op.concat(args[0], axis=axis))

    def _chunk(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        chunks = node.args[1]
        # TODO below might just pass none if node.args <= 2 ?
        dim = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim", 0)
        name = node.args[3] if len(node.args) > 3 else node.kwargs.get("name", None)
        return self.block_builder.emit(relax.op.chunk(x, chunks, dim, name))

    def _cumsum(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]

        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", None)
        if "dtype" in node.kwargs:
            dtype = self._convert_data_type(str(node.kwargs["dtype"]), self.env)
        else:
            dtype = None
        if "out" in node.kwargs:
            raise ValueError("specifying out for cumsum is not supported yet")

        return self.block_builder.emit(relax.op.cumsum(x, dim, dtype))

    def _expand(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        sizes = args[1:] if len(args) > 2 else args[1]
        broadcast_shape, in_shape = [], self.shape_of(args[0])
        for idx, i in enumerate(sizes):
            if isinstance(i, int) and i == -1:
                broadcast_shape.append(in_shape[idx])
            else:
                broadcast_shape.append(i)
        return self.block_builder.emit(relax.op.broadcast_to(args[0], broadcast_shape))

    def _expand_as(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        # args[0] -> 'self' tensor
        # args[1] -> 'other' tensor
        data = args[0]
        other_shape = self.shape_of(args[1])  # This is the shape of 'other'
        return self.block_builder.emit(relax.op.broadcast_to(data, other_shape))

    def _flip(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dims = node.args[1] if len(node.args) > 1 else node.kwargs.get("dims", None)
        if isinstance(dims, (list, tuple)) and len(dims) > 0:
            dims = dims[0]
        elif not isinstance(dims, int):
            raise TypeError(f"flip expects an integer axis, but got {type(dims)}: {dims}")
        return self.block_builder.emit(relax.op.flip(x, dims))

    def _gather(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        index = self.env[node.args[2]]
        return self.block_builder.emit(relax.op.gather_elements(x, index, axis=dim))

    def _permute(self, node: fx.Node) -> relax.Var:
        import torch  # type: ignore

        args = self.retrieve_args(node)
        x = args[0]
        dims = args[1] if isinstance(args[1], (torch.Size, tuple, list)) else args[1:]
        return self.block_builder.emit(relax.op.permute_dims(x, dims))

    def _repeat(self, node: fx.Node) -> relax.Var:
        import torch  # type: ignore

        args = self.retrieve_args(node)
        x = args[0]
        dims = args[1] if isinstance(args[1], (torch.Size, tuple, list)) else args[1:]
        return self.block_builder.emit(relax.op.tile(x, dims))

    def _reshape(self, node: fx.Node) -> relax.Var:
        import torch  # type: ignore

        args = self.retrieve_args(node)
        x = args[0]
        dims = args[1] if isinstance(args[1], (torch.Size, tuple, list)) else args[1:]
        return self.block_builder.emit(relax.op.reshape(x, dims))

    def _scatter(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        if len(node.args) == 1:
            dim = node.kwargs["dim"]
            index = self.env[node.kwargs["index"]]
            src = self.env[node.kwargs["src"]]
        elif len(node.args) == 4:
            dim = node.args[1]
            index = self.env[node.args[2]]
            src = self.env[node.args[3]]
        else:
            raise Exception("Unexpected args " + str(node.args))
        return self.block_builder.emit(relax.op.scatter_elements(x, index, src, axis=dim))

    def _split(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        split_size = node.args[1]
        dim = node.args[2] if len(node.args) > 2 else node.kwargs.get("dim", 0)
        if isinstance(split_size, (list, tuple)):
            n_section = []
            for s in split_size[:-1]:
                cum_sum = 0 if not n_section else n_section[-1]
                n_section.append(s + cum_sum)
        else:
            n_section = (self.shape_of(x)[dim].value + split_size - 1) // split_size
        return self.block_builder.emit(relax.op.split(x, n_section, dim))

    def _squeeze(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dim = node.args[1] if len(node.args) > 1 else node.kwargs.get("dim", None)
        return self.block_builder.emit(relax.op.squeeze(x, dim))

    def _stack(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        axis = args[1] if len(node.args) > 1 else node.kwargs.get("dim", 0)
        in_args = args[0]
        assert all(
            a.struct_info.shape[axis] == in_args[0].struct_info.shape[axis] for a in in_args[1:]
        ), "Expect all dim at {} to be the same, get {}".format(
            axis, [a.struct_info.shape for a in args]
        )
        cat = self.block_builder.emit(relax.op.concat(in_args, axis=axis))
        s_shape = []
        for idx, s in enumerate(cat.struct_info.shape):
            if idx == axis:
                s_shape.extend([len(in_args), in_args[0].struct_info.shape[axis]])
            else:
                s_shape.append(s)
        return self.block_builder.emit(relax.op.reshape(cat, s_shape))

    def _take(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        indices = self.env[node.args[1]]
        indices = self.block_builder.emit(relax.op.astype(indices, "int32"))
        return self.block_builder.emit(relax.op.take(x, indices))

    def _tile(self, node: fx.Node) -> relax.Var:
        import torch  # type: ignore

        args = self.retrieve_args(node)
        x = args[0]
        dims = args[1] if isinstance(args[1], (torch.Size, tuple, list)) else args[1:]
        return self.block_builder.emit(relax.op.tile(x, dims))

    def _transpose(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        full_idx = list(range(len(self.shape_of(args[0]))))
        full_idx[args[1]], full_idx[args[2]] = full_idx[args[2]], full_idx[args[1]]
        return self.block_builder.emit(relax.op.permute_dims(args[0], full_idx))

    ########## Creation ##########

    def _detach(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        # if len(node.args) == 2:
        #     if isinstance(node.args[1], torch.dtype):
        #         dtype = self._convert_data_type(node.args[1], self.env)
        #         return self.block_builder.emit(relax.op.astype(x, dtype))
        # elif "dtype" in node.kwargs:
        #     dtype = self._convert_data_type(node.kwargs["dtype"], self.env)
        #     return self.block_builder.emit(relax.op.astype(x, dtype))
        return x

    def _to_copy(self, node: fx.Node) -> relax.Var:
        import torch  # type: ignore

        x = self.env[node.args[0]]
        if len(node.args) == 2:
            if isinstance(node.args[1], torch.dtype):
                dtype = self._convert_data_type(node.args[1], self.env)
                return self.block_builder.emit(relax.op.astype(x, dtype))
        elif "dtype" in node.kwargs:
            dtype = self._convert_data_type(node.kwargs["dtype"], self.env)
            return self.block_builder.emit(relax.op.astype(x, dtype))
        return x

    def _arange(self, node: fx.Node) -> relax.Var:
        import torch  # type: ignore

        start_end_step = [None, None, None]
        if "start" in node.kwargs:
            start_end_step[0] = node.kwargs["start"]
        if "end" in node.kwargs:
            start_end_step[1] = node.kwargs["end"]
        if "step" in node.kwargs:
            start_end_step[2] = node.kwargs["step"]

        if len(node.args) == 1:
            assert start_end_step[1] is None
            start_end_step[1] = node.args[0]
        elif len(node.args) == 2:
            assert start_end_step[0] is None
            assert start_end_step[1] is None
            start_end_step[0] = node.args[0]
            start_end_step[1] = node.args[1]
        elif len(node.args) == 3:
            assert start_end_step[0] is None
            assert start_end_step[1] is None
            assert start_end_step[2] is None
            start_end_step[0] = node.args[0]
            start_end_step[1] = node.args[1]
            start_end_step[2] = node.args[2]

        if start_end_step[0] is None:
            start_end_step[0] = 0
        if start_end_step[2] is None:
            start_end_step[2] = 1

        if "dtype" in node.kwargs:
            dtype = self._convert_data_type(str(node.kwargs["dtype"]), self.env)
        elif any([isinstance(x, float) for x in start_end_step]):
            dtype = self._convert_data_type(torch.get_default_dtype())
        else:
            dtype = "int64"
        start_end_step = [
            self.env[x] if isinstance(x, torch.fx.Node) else x for x in start_end_step
        ]
        return self.block_builder.emit(relax.op.arange(*start_end_step, dtype=dtype))

    def _empty(self, node: fx.Node) -> relax.Var:
        dtype = self._convert_data_type(str(node.kwargs["dtype"]), self.env)
        return self.block_builder.emit(relax.op.zeros(node.args[0], dtype))

    def _fill(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        x = args[0]
        dtype = x.struct_info.dtype
        value = args[1] if isinstance(args[1], relax.Expr) else relax.const(args[1], dtype)
        return self.block_builder.emit(relax.op.full(x.struct_info.shape, value, dtype))

    def _new_ones(self, node: fx.Node) -> relax.Var:
        args = self.retrieve_args(node)
        self_var = args[0]
        size = args[1] if isinstance(args[1], (list, tuple)) else args[1:]
        if not isinstance(size, (list, tuple)):
            size = (size,)
        size = relax.ShapeExpr(size)
        return self.block_builder.emit(
            relax.op.full(
                size,
                relax.const(1, self_var.struct_info.dtype),
                self_var.struct_info.dtype,
            )
        )

    ########## Others ##########

    def _getitem(self, node: fx.Node) -> relax.Var:
        import torch

        x = self.env[node.args[0]]
        if isinstance(x, (list, tuple, relax.ShapeExpr, relax.Tuple)):
            return x[node.args[1]]
        elif isinstance(x, relax.Var):
            if isinstance(x.struct_info, relax.TupleStructInfo):
                return self.block_builder.emit(relax.TupleGetItem(x, node.args[1]))

            assert isinstance(x.struct_info, relax.TensorStructInfo)
            take_indices = []
            take_axes = []
            stride_begin = []
            stride_end = []
            stride = []
            stride_axes = []
            expand_dim = []
            i = 0
            shape = self.shape_of(x)
            non_ellipsis_cnt = 0
            for index in node.args[1]:
                if isinstance(index, (int, slice, torch.fx.Node)):
                    non_ellipsis_cnt += 1
            for index in node.args[1]:
                if isinstance(index, int):
                    stride_begin.append(index)
                    stride_end.append(index + 1)
                    stride.append(1)
                    stride_axes.append(i)
                    i = i + 1
                elif isinstance(index, slice):
                    stride_begin.append(0 if index.start is None else index.start)
                    stride_end.append(shape[i] if index.stop is None else index.stop)
                    stride.append(1 if index.step is None else index.step)
                    stride_axes.append(i)
                    i = i + 1
                elif index is None:
                    expand_dim.append(len(stride_axes) + len(expand_dim))
                elif index is Ellipsis:
                    for _ in range(len(shape) - non_ellipsis_cnt):
                        stride_begin.append(0)
                        stride_end.append(shape[i])
                        stride.append(1)
                        stride_axes.append(i)
                        i += 1
                elif isinstance(index, torch.fx.Node):
                    node_index = self.env[index]
                    if not isinstance(node_index, relax.Expr):
                        raise ValueError(
                            "Unsupported index type for relax.op.take: " + str(type(node_index))
                        )
                    take_indices.append(node_index)
                    take_axes.append(i)
                    i = i + 1
                else:
                    raise ValueError("Unsupported index type: " + str(type(index)))
            while i < len(shape):
                stride_begin.append(0)
                stride_end.append(shape[i])
                stride.append(1)
                stride_axes.append(i)
                i += 1
            taken = x
            if len(take_indices) > 1:
                raise ValueError("Multiple tensors as index not yet supported")
            for each_index, each_axis in zip(take_indices, take_axes):
                taken = self.block_builder.emit(relax.op.take(taken, each_index, each_axis))
            sliced = self.block_builder.emit(
                relax.op.strided_slice(taken, stride_axes, stride_begin, stride_end, stride)
            )
            sliced_shape = list(self.shape_of(sliced))
            for i in expand_dim:
                sliced_shape.insert(i, 1)
            return self.block_builder.emit(relax.op.reshape(sliced, sliced_shape))
        elif isinstance(x, relax.Constant):
            dtype = x.struct_info.dtype
            return relax.const(x.data.numpy()[node.args[1]], dtype)
        else:
            assert False

    @abc.abstractmethod
    def create_convert_map(
            self,
    ) -> Dict[Union[torch.nn.Module, str], Callable[[fx.Node], relax.Var]]:
        """Create convert map"""


from torch import nn
import numpy as np
import torch
from torch.export import export
from torch.nn import Softmax

# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=invalid-name, inconsistent-return-statements, unidiomatic-typecheck
# pylint: disable=import-outside-toplevel
"""PyTorch ExportedProgram of Relax."""
from collections import ChainMap, OrderedDict
from functools import partial
from typing import Callable, Dict, List, Tuple

import torch
import tvm
from tvm import relax


class ExportedProgramImporter(BaseFXGraphImporter):
    """An importer from ExportedProgram to Relax."""

    from torch import fx

    ########## Unary Ops ##########

    def _hardtanh(self, node: fx.Node) -> relax.Expr:
        args = self.retrieve_args(node)
        x = args[0]
        min_val = node.args[1] if len(args) > 1 else node.kwargs("min_val", -1.0)
        max_val = node.args[2] if len(args) > 2 else node.kwargs("max_val", 1.0)
        return self.block_builder.emit(relax.op.clip(x, min_val, max_val))

    ########## Neural Network ##########

    def _batch_norm_legit_no_training(self, node: fx.Node) -> relax.Var:
        import numpy as np

        x = self.env[node.args[0]]
        channel = int(self.shape_of(x)[1])
        dtype = x.struct_info.dtype
        weight = self.env.get(node.args[1], relax.const(np.ones(channel), dtype=dtype))
        bias = self.env.get(node.args[2], relax.const(np.zeros(channel), dtype=dtype))
        running_mean = self.env.get(node.args[3], relax.const(np.zeros(channel), dtype=dtype))
        running_var = self.env.get(node.args[4], relax.const(np.ones(channel), dtype=dtype))
        momentum = node.args[5] if len(node.args) > 5 else node.kwargs.get("momentum", 0.1)
        eps = node.args[6] if len(node.args) > 6 else node.kwargs.get("eps", 1e-05)

        return self.block_builder.emit(
            relax.op.nn.batch_norm(
                x,
                weight,
                bias,
                running_mean,
                running_var,
                axis=1,
                epsilon=eps,
                momentum=momentum,
            )
        )

    def _group_norm(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        num_groups = node.args[1]
        gamma = self.env[node.args[2]] if len(node.args) > 2 else None
        beta = self.env[node.args[3]] if len(node.args) > 3 else None
        eps = node.args[4] if len(node.args) > 4 else 1e-05

        dim = len(self.shape_of(x))
        return self.block_builder.emit(
            relax.op.nn.group_norm(
                x,
                gamma,
                beta,
                num_groups=num_groups,
                channel_axis=1,
                axes=list(range(2, dim)),
                epsilon=eps,
            )
        )

    def _batch_norm(self, node: fx.Node) -> relax.Var:
        # print("PRINTING BATCH NORM")
        # print("type(node): ",type(node))
        # print("node: ",node)
        # print("type(node.args): ",type(node.args))
        # print("node.args: ",node.args)
        # print("type(node.args[0]): ",type(node.args[0]))
        # print("node.args[0]: ",node.args[0])
        # print("DONE PRINTING BATCH NORM")

        x = self.env[node.args[0]]
        gamma = self.env[node.args[1]]
        beta = self.env[node.args[2]]
        moving_mean = self.env[node.args[3]]
        moving_var = self.env[node.args[4]]
        center = node.args[5] if len(node.args) > 5 else None  # TODO copilot suggests node.kwargs.get("center", True)]
        momentum = node.args[6] if len(node.args) > 6 else None  # TODO copilot says node.kwargs.get("momentum", 0.1)
        eps = node.args[7] if len(node.args) > 4 else 1e-05
        scale = node.args[8] if len(node.args) > 8 else None

        return self.block_builder.emit(
            # adds the line of relax function into the ir function  (and produces left hand side)
            relax.op.nn.batch_norm(
                # creates RHS of a line in a relax fonction (starts with R.Tensor... or R.call_tir...)
                data=x,
                gamma=gamma,
                beta=beta,
                moving_mean=moving_mean,
                moving_var=moving_var,
                axis=1,  # TODO ?????
                epsilon=eps,
                center=center,
                scale=scale,
                momentum=momentum
            )[0]  # ruihang prefers
        )

    def _upsample_impl(
            self, x: relax.Expr, size, align_corners: bool, scale_factor, method: str
    ) -> relax.Var:
        coord_trans = "align_corners" if align_corners else "half_pixel"

        if size is None:
            shape = self.shape_of(x)
            assert isinstance(shape, relax.ShapeExpr)
            if isinstance(scale_factor, (tuple, list)):
                assert len(scale_factor) == len(shape) - 2
                size = tuple(
                    int(shape[i].value * scale_factor[i - 2]) for i in range(2, len(shape))
                )
            else:
                size = tuple(int(shape[i].value * scale_factor) for i in range(2, len(shape)))

        return self.block_builder.emit(
            relax.op.image.resize2d(
                x, size, layout="NCHW", method=method, coordinate_transformation_mode=coord_trans
            )
        )

    def _upsample_bilinear2d(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        size = node.args[1] if len(node.args) > 1 else node.kwargs.get("size", None)
        align_corners = (
            node.args[2] if len(node.args) > 2 else node.kwargs.get("align_corners", True)
        )
        scale_factor = node.args[3] if len(node.args) > 3 else node.kwargs.get("scale_factor",
                                                                               1)  # TODO non-sense to default to 1! Understand why "None" doesn't work!
        return self._upsample_impl(x, size, align_corners, scale_factor, "linear")

    def _upsample_nearest2d(self, node: fx.node) -> relax.Var:
        x = self.env[node.args[0]]
        size = node.args[1] if len(node.args) > 1 else node.kwargs.get("size", None)
        align_corners = (
            node.args[2] if len(node.args) > 2 else node.kwargs.get("align_corners", True)
        )
        if len(node.args) > 3:
            print("AAAAAAAAAAAAAAAA")
            scale_factor = node.args[3]
        else:
            print("BBBBBBBBBBBBBBBB")
            print("the kwargs are: ", list(node.kwargs.keys()))
            scale_factor = node.kwargs.get("scale_factor",
                                           1)  # TODO non-sense to default to 1! Understand why "None" doesn't work!

        print("size: ", size)
        print("align_corners: ", align_corners)
        print("scale_factor: ", scale_factor)
        scale_factor = 2
        return self._upsample_impl(x, size, align_corners, scale_factor, "nearest_neighbor")

    ########## Manipulation ##########

    def _select(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        dim = node.args[1]
        index = relax.const(node.args[2], "int64")
        return self.block_builder.emit(relax.op.take(x, index, dim))

    def _slice(self, node: fx.Node) -> relax.Var:
        x = self.env[node.args[0]]
        axes = [node.args[1]]
        begin = [node.args[2]]
        end = [node.args[3]]
        stride = [node.args[4] if len(node.args) > 4 else 1]
        return self.block_builder.emit(relax.op.strided_slice(x, axes, begin, end, stride))

    ########## Others ##########

    def create_convert_map(
            self,
    ) -> Dict[str, Callable[[fx.Node], relax.Var]]:
        import operator

        return {
            # unary
            "abs.default": self._unary_op(relax.op.abs),
            "acos.default": self._unary_op(relax.op.acos),
            "acosh.default": self._unary_op(relax.op.acosh),
            "asin.default": self._unary_op(relax.op.asin),
            "asinh.default": self._unary_op(relax.op.asinh),
            "atan.default": self._unary_op(relax.op.atan),
            "atanh.default": self._unary_op(relax.op.atanh),
            "bitwise_not.default": self._unary_op(relax.op.bitwise_not),
            "ceil.default": self._unary_op(relax.op.ceil),
            "clamp.default": self._clamp,
            "clamp_min.default": self._clamp_min,
            "cos.default": self._unary_op(relax.op.cos),
            "cosh.default": self._unary_op(relax.op.cosh),
            "dropout.default": lambda node: self.env[node.args[0]],
            "erf.default": self._unary_op(relax.op.erf),
            "exp.default": self._unary_op(relax.op.exp),
            "floor.default": self._unary_op(relax.op.floor),
            "gelu.default": self._gelu,
            "hardsigmoid.default": self._hardsigmoid,
            "hardswish.default": self._hardswish,
            "hardtanh.default": self._hardtanh,
            "isfinite.default": self._unary_op(relax.op.isfinite),
            "isinf.default": self._unary_op(relax.op.isinf),
            "isnan.default": self._unary_op(relax.op.isnan),
            "leaky_relu.default": self._leakyrelu,
            "log.default": self._unary_op(relax.op.log),
            "log_softmax.int": self._log_softmax,
            "neg.default": self._unary_op(relax.op.negative),
            "relu.default": self._unary_op(relax.op.nn.relu),
            "round.default": self._round,
            "rsqrt.default": self._unary_op(relax.op.rsqrt),
            "sigmoid.default": self._unary_op(relax.op.sigmoid),
            "sign.default": self._unary_op(relax.op.sign),
            "silu.default": self._unary_op(relax.op.nn.silu),
            "sin.default": self._unary_op(relax.op.sin),
            "sinh.default": self._unary_op(relax.op.sinh),
            "softmax.int": self._softmax,
            "sqrt.default": self._unary_op(relax.op.sqrt),
            "square.default": self._unary_op(relax.op.square),
            "tan.default": self._unary_op(relax.op.tan),
            "tanh.default": self._unary_op(relax.op.tanh),
            "tril.default": self._tril_triu(relax.op.tril),
            "triu.default": self._tril_triu(relax.op.triu),
            # binary
            "add.Tensor": self._binary_op(relax.op.add, operator.add),
            "div.Tensor": self._binary_op(relax.op.divide, operator.truediv),
            "eq.Scalar": self._binary_op(relax.op.equal, operator.eq),
            "eq.Tensor": self._binary_op(relax.op.equal, operator.eq),
            "floor_divide.default": self._binary_op(relax.op.floor_divide, operator.floordiv),
            "ge.Scalar": self._binary_op(relax.op.greater_equal, operator.ge),
            "ge.Tensor": self._binary_op(relax.op.greater_equal, operator.ge),
            "gt.Scalar": self._binary_op(relax.op.greater, operator.gt),
            "gt.Tensor": self._binary_op(relax.op.greater, operator.gt),
            "le.Scalar": self._binary_op(relax.op.less_equal, operator.le),
            "le.Tensor": self._binary_op(relax.op.less_equal, operator.le),
            "lt.Scalar": self._binary_op(relax.op.less, operator.lt),
            "lt.Tensor": self._binary_op(relax.op.less, operator.lt),
            "matmul.default": self._binary_op(
                partial(relax.op.linear_algebra.matmul, out_dtype="float32"), operator.matmul
            ),
            "max.other": self._binary_op(relax.op.maximum, max),
            "min.other": self._binary_op(relax.op.minimum, min),
            "remainder.Tensor": self._binary_op(mod, operator.mod),
            "remainder.Scalar": self._binary_op(mod, operator.mod),
            "mul.Tensor": self._binary_op(relax.op.multiply, operator.mul),
            "ne.Tensor": self._binary_op(relax.op.not_equal, operator.ne),
            "ne.Scalar": self._binary_op(relax.op.not_equal, operator.ne),
            "pow.Tensor_Scalar": self._binary_op(relax.op.power, operator.pow),
            "pow.Tensor_Tensor": self._binary_op(relax.op.power, operator.pow),
            "sub.Tensor": self._binary_op(relax.op.subtract, operator.sub),
            "__and__.Tensor": self._binary_op(relax.op.bitwise_and, operator.and_),
            "__and__.Scalar": self._binary_op(relax.op.bitwise_and, operator.and_),
            "__or__.Tensor": self._binary_op(relax.op.bitwise_or, operator.or_),
            "__or__.Scalar": self._binary_op(relax.op.bitwise_or, operator.or_),
            "__xor__.Tensor": self._binary_op(relax.op.bitwise_xor, operator.xor),
            "__xor__.Scalar": self._binary_op(relax.op.bitwise_xor, operator.xor),
            # linear algebra
            "linalg_vector_norm.default": self._linalg_vector_norm,
            # neural network
            "_native_batch_norm_legit_no_training.default": self._batch_norm_legit_no_training,
            "adaptive_avg_pool2d.default": self._adaptive_avg_pool2d,
            "addmm.default": self._addmm,
            "avg_pool2d.default": self._avg_pool2d,
            "baddbmm.default": self._baddbmm,
            "bmm.default": self._binary_op(
                partial(relax.op.linear_algebra.matmul, out_dtype="float32"), operator.matmul
            ),
            "conv_transpose1d.default": self._conv_transpose1d,
            "conv_transpose2d.input": self._conv_transpose2d,
            "conv1d.default": self._conv1d,
            "conv2d.default": self._conv2d,
            "conv3d.default": self._conv3d,
            "einsum.default": self._einsum,
            "embedding.default": lambda node: self._embedding_impl(
                self.env[node.args[1]], self.env[node.args[0]]
            ),
            "group_norm.default": self._group_norm,
            "batch_norm.default": self._batch_norm,
            "layer_norm.default": self._layer_norm,
            "linear.default": self._linear,
            "max_pool2d.default": self._max_pool2d,
            "scaled_dot_product_attention.default": self._scaled_dot_product_attention,
            "unbind.int": self._unbind,
            "upsample_bilinear2d.vec": self._upsample_bilinear2d,
            "upsample_nearest2d.vec": self._upsample_nearest2d,
            # statistical
            "mean.dim": self._mean,
            "sum.dim_IntList": self._sum,
            # search
            "argmax.default": self._argmax_argmin(relax.op.argmax),
            "argmin.default": self._argmax_argmin(relax.op.argmin),
            # tensor manipulation
            "cat.default": self._cat,
            "chunk.default": self._chunk,
            "concat.default": self._cat,
            "cumsum.default": self._cumsum,
            "expand.default": self._expand,
            "expand_as.default": self._expand_as,
            "permute.default": self._permute,
            "repeat.default": self._repeat,
            "reshape.default": self._reshape,
            "select.int": self._select,
            "slice.Tensor": self._slice,
            "split.Tensor": self._split,
            "squeeze.default": self._squeeze,
            "squeeze.dim": self._squeeze,
            "tile.default": self._tile,
            "transpose.int": self._transpose,
            "unsqueeze.default": lambda node: self.block_builder.emit(
                relax.op.expand_dims(self.env[node.args[0]], node.args[1])
            ),
            "view.default": self._reshape,
            "reshape.default": self._reshape,
            # tensor creation
            "_to_copy.default": self._to_copy,
            "lift_fresh_copy.default": self._to_copy,
            "detach_.default": self._detach,
            "arange.start": self._arange,
            "contiguous.default": lambda node: self.env[node.args[0]],
            # TODO would be more efficient to not always do this, and only do it if not contiguous, but need to find a way to check
            "clone.default": lambda node: self.env[node.args[0]],
            "empty.memory_format": self._empty,
            "fill.Scalar": self._fill,
            "new_ones.default": self._new_ones,
            # other
            "getitem": self._getitem,
        }

    def create_input_vars(
            self, exported_program: torch.export.ExportedProgram
    ) -> Tuple[Dict[str, relax.Var], Dict[str, relax.Var]]:
        """Create relax input vars."""
        parameters_buffers_constants = OrderedDict()
        user_inputs = OrderedDict()
        for spec in exported_program.graph_signature.input_specs:
            name_hint = spec.arg.name
            if spec.kind is torch.export.graph_signature.InputKind.CONSTANT_TENSOR:
                shape = exported_program.tensor_constants[spec.target].shape
                torch_dtype = exported_program.tensor_constants[spec.target].dtype
            elif spec.kind is torch.export.graph_signature.InputKind.USER_INPUT:
                for node in exported_program.graph.find_nodes(op="placeholder", target=spec.target):
                    if node.name == name_hint:
                        shape = node.meta["tensor_meta"].shape
                        torch_dtype = node.meta["tensor_meta"].dtype
                        break
            else:
                # PARAMETER or BUFFER
                shape = exported_program.state_dict[spec.target].shape
                torch_dtype = exported_program.state_dict[spec.target].dtype

            dtype = self._convert_data_type(torch_dtype)
            relax_var = relax.Var(name_hint, relax.TensorStructInfo(shape, dtype))
            if spec.kind is torch.export.graph_signature.InputKind.USER_INPUT:
                user_inputs[name_hint] = relax_var
            else:
                parameters_buffers_constants[name_hint] = relax_var

        return parameters_buffers_constants, user_inputs

    def from_exported_program(
            self,
            exported_program: torch.export.ExportedProgram,
            keep_params_as_input: bool,
            unwrap_unit_return_tuple: bool,
            no_bind_return_tuple: bool,
    ) -> tvm.IRModule:
        """Convert a PyTorch ExportedProgram to a Relax program."""
        from torch import fx  # type: ignore

        # Create input variables.
        parameter_buffer_constant_vars, user_input_vars = self.create_input_vars(exported_program)
        inputs_vars = user_input_vars.copy()
        inputs_vars.update(parameter_buffer_constant_vars)

        # Initialize the block builder with a function and a dataflow block.
        self.block_builder = relax.BlockBuilder()
        func_name = "main"
        func_attrs = {"num_input": len(user_input_vars)} if keep_params_as_input else None

        nodes: List[fx.Node] = exported_program.graph.nodes
        with self.block_builder.function(
                name=func_name, params=list(inputs_vars.values()).copy(), attrs=func_attrs
        ):
            output = None
            with self.block_builder.dataflow():
                # Translate the model.
                for node in nodes:
                    # print("the node has type: ",type(node))
                    # print(node)
                    if node.op == "placeholder":
                        if "grapharg" in node.meta and node.meta["grapharg"].fake_tensor is None:
                            # Ignore sym input
                            continue

                        self.env[node] = inputs_vars[node.name]
                    elif node.op == "output":
                        args = self.retrieve_args(node)
                        assert len(args) == 1
                        assert isinstance(args[0], (tuple, relax.Tuple))

                        if unwrap_unit_return_tuple and len(args[0]) == 1:
                            output = self.block_builder.emit_output(args[0][0])
                        elif no_bind_return_tuple:
                            output = []
                            for ret in args[0]:
                                output.append(self.block_builder.emit_output(ret))
                        else:
                            output = self.block_builder.emit_output(args[0])
                        break
                    elif node.op == "get_attr":
                        self.env[node] = getattr(exported_program.graph_module, node.target)
                    elif node.op == "call_function":
                        func_name = node.target.__name__
                        print("Found a func_name of ", func_name)
                        assert (
                                func_name in self.convert_map
                        ), f"Unsupported function type {func_name}"
                        self.env[node] = self.convert_map[func_name](node)
                    else:
                        raise ValueError(f"Unsupported op {node.op}")
            assert output is not None
            self.block_builder.emit_func_output(output)

        to_bind_parameters = ChainMap(
            OrderedDict(exported_program.named_buffers()), exported_program.constants
        )
        if not keep_params_as_input:
            to_bind_parameters = to_bind_parameters.new_child(
                OrderedDict(exported_program.named_parameters())
            )

        binding = {}
        for tensor_name, tensor_value in to_bind_parameters.items():
            # find relax var name from graph signature
            for spec in exported_program.graph_signature.input_specs:
                if tensor_name == spec.target:
                    bind_name = spec.arg.name
                    break
            binding[bind_name] = tvm.nd.from_dlpack(tensor_value.detach())

        mod = self.block_builder.get()
        mod = relax.transform.BindParams("main", binding)(mod)

        if keep_params_as_input:
            parameters = dict(exported_program.named_parameters())
            params = [tvm.nd.from_dlpack(p.detach()) for p in parameters.values()]
            mod["main"] = mod["main"].with_attr("params", params)

        return mod


def from_exported_program(
        exported_program: torch.export.ExportedProgram,
        *,
        keep_params_as_input: bool = False,
        unwrap_unit_return_tuple: bool = False,
        no_bind_return_tuple: bool = False,
) -> tvm.IRModule:
    """Convert a PyTorch ExportedProgram to a Relax program

    Parameters
    ----------
    exported_program : torch.export.ExportedProgram
        The PyTorch ExportedProgram to convert.

    keep_params_as_input : bool
        Whether to keep model parameters as input variables.

    unwrap_unit_return_tuple : bool
        A boolean flag indicating if to the return value when it is an unit tuple.
        When the return value is not a unit tuple, no unwrap will take place.

    no_bind_return_tuple : bool
        A boolean flag indicating whether to bind the return tuple as a relax var.
        If the flag is true and the return value is a tuple, it will not bind it to a var.

    Returns
    -------
    output : tvm.IRModule
        The import result IRModule, with the function "main" containing the
        translated logic.

    Examples
    --------
    Users can use the torch.export.export() to extract a torch.export.ExportedProgram
    from a PyTorch model. The following codes show how to convert a PyTorch model to
    a Relax program.

    .. code-block:: python

        # Import the importer.
        import tvm
        from tvm.relax.frontend.torch import from_exported_program
        import torch
        from torch.export import export

        # Define the module
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(in_features=10, out_features=7, bias=True)

            def forward(self, input):
                return self.linear(input)

        # Instantiate the model and create the input info dict.
        torch_model = MyModule()

        # Use torch.export.export() to convert the PyTorch model into ExportedProgram.
        example_args = (torch.rand(128, 10, dtype=torch.float32),)
        exported_program = export(torch_model, args=example_args)

        # Use the importer to import the ExportedProgram to Relax.
        mod: tvm.IRModule = from_exported_program(exported_program)
    """
    # decompose into Core ATen operators
    exported_program.run_decompositions()

    return ExportedProgramImporter().from_exported_program(
        exported_program,
        keep_params_as_input,
        unwrap_unit_return_tuple,
        no_bind_return_tuple,
    )


torch_model = Softmax(dim=2).eval()

raw_data = np.random.rand(1, 4, 32, 4096).astype("float32")

torch_data = torch.from_numpy(raw_data)

# Give an example argument to torch.export
example_args = (torch_data,)

# Convert the model to IRModule
# TODO what does , unwrap_unit_return_tuple=True do? should we include?
with torch.no_grad():
    exported_program = export(torch_model, example_args)
    mod_from_torch = from_exported_program(
        exported_program, keep_params_as_input=True  # , unwrap_unit_return_tuple=True
    )

tvm_mod, tvm_params = relax.frontend.detach_params(mod_from_torch)
tvm_mod.show()


