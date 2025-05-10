from typing import Tuple

import torch
from torch import Tensor


@torch.library.custom_op("mylib::pyattn", mutates_args=())
def attn_in_py(q: Tensor, k: Tensor, v: Tensor) -> Tuple[Tensor, Tensor]:
    x = torch.matmul(q, k.transpose(0, 1))
    a = torch.tanh(x)
    o = torch.matmul(a, v)
    return o, a


def backward(ctx, grad1, grad2):
    # We are lucky that the operator definition returns the required intermediate data `a`
    print(grad1.shape, grad2.shape)
    q, k, v = ctx.inputs
    o, a = ctx.outputs
    tanh_grad = (1 - a**2)
    expr1 = (grad1 @ v.T) * tanh_grad
    q_grad = expr1 @ k
    k_grad = expr1.T @ q
    v_grad = a.T @ grad1

    expr2 = (grad2 * tanh_grad)
    q_grad += expr2 @ k
    k_grad += expr2.T @ q
    v_grad += 0

    return q_grad, k_grad, v_grad


def setup_context(ctx, inputs, output):
    ctx.inputs = inputs
    ctx.outputs = output


attn_in_py.register_autograd(backward, setup_context=setup_context)
