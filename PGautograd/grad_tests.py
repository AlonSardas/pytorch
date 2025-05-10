from typing import Tuple

import myattnop
import torch
from pyattndef import Exp2, MyAttnOp
from torch import Tensor
from torch.autograd import gradcheck


def attn(q, k, v):
    x = torch.matmul(q, k.transpose(0, 1))
    a = torch.tanh(x)
    o = torch.matmul(a, v)
    return o, a


def test_attn_grads():
    q = torch.rand(2, 3, requires_grad=True)
    k = torch.rand(2, 3, requires_grad=False)
    v = torch.rand(2, 4, requires_grad=False)

    xx = torch.matmul(q, k.transpose(0, 1))

    o, a = attn(q, k, v)
    L_o = o.diag().sum()
    L_o.backward()
    print(q.grad)
    # Not sure what's the best way to achieve this. But using new axis and indices i,n,j should work
    I, J = xx.shape[0], k.shape[1]
    # N = k.shape[0]
    # print(I, N, J)
    # v_t_fs = v.T[:I, :].unsqueeze(2).expand(I, N, J)
    # x_fs = (1/(xx.cosh()**2)).unsqueeze(2).expand(I, N, J)
    # k_fs = k.unsqueeze(0).expand(I, N, J)
    # expected = torch.sum(v_t_fs * x_fs * k_fs, 1)
    expected = (v.T[:I, :] / (xx.cosh() ** 2)) @ k
    print(expected)


# See https://docs.pytorch.org/docs/stable/notes/autograd.html
# Define a train function to be used in different threads
def train_fn():
    x = torch.ones(5, 5, requires_grad=True)
    # forward
    y = (x + 3) * (x + 4) * 0.5
    # backward
    y.sum().backward()
    # potential optimizer update
    print(x.grad)


def test_sum_diag():
    x = torch.ones(5, 5, requires_grad=True)
    L = x.diag().sum()
    L.backward()
    # potential optimizer update
    print(x.grad)


def test_pyattn_op():
    # torch
    q = torch.rand(2, 3, requires_grad=True)
    k = torch.rand(2, 3, requires_grad=True)
    v = torch.rand(2, 4, requires_grad=True)
    o, a = torch.ops.mylib.pyattn(q, k, v)
    L_o = o.diag().sum()
    L_o.backward()
    print_grads(*_calc_grads(q, k, v))

    # o, a = torch.ops.mylib.pyattn(q, k, v)
    # L_a = a.diag().sum()
    # L_a.backward()
    # print(a.grad)

    # print(q, k , v)
    print_grads(*_calc_grads(q, k, v))

    print("Actual check with gradcheck")
    input_tensors = (
        torch.randn(20, 30, dtype=torch.double, requires_grad=True),
        torch.randn(20, 30, dtype=torch.double, requires_grad=True),
        torch.randn(20, 40, dtype=torch.double, requires_grad=True),
    )
    # test = gradcheck(torch.ops.mylib.pyattn, input_tensors, eps=1e-6, atol=1e-4)
    # print(test)


def test_Exp_op():
    q = torch.rand(2, 3, requires_grad=True)
    out = Exp2.apply(q)
    L_o = out.diag().sum()
    L_o.backward()
    print(q.grad)


def test_my_attn_op():
    q = torch.rand(2, 3, requires_grad=True)
    k = torch.rand(2, 3, requires_grad=False)
    v = torch.rand(2, 4, requires_grad=False)

    o, a = MyAttnOp.apply(q, k, v)
    L_o = o.diag().sum()
    L_o.backward()
    print(q.grad)


def test_nat_attn():
    q = torch.rand(2, 3, requires_grad=True)
    k = torch.rand(2, 3, requires_grad=True)
    v = torch.rand(2, 4, requires_grad=True)
    o, a = torch.nat_attn(q, k, v)
    L_o = o.diag().sum()
    L_o.backward()
    print_grads(q, k, v)
    print_grads(*_calc_grads(q, k, v))

    print("Actual check with gradcheck")
    input_tensors = (
        torch.randn(20, 30, dtype=torch.double, requires_grad=True),
        torch.randn(20, 30, dtype=torch.double, requires_grad=True),
        torch.randn(20, 40, dtype=torch.double, requires_grad=True),
    )
    op = lambda *args: torch.nat_attn(*args)[0]
    test = gradcheck(op, input_tensors, eps=1e-6, atol=1e-4)
    print(test)



def _calc_grads(q, k, v: Tensor):
    if q.grad is not None:
        q.grad.zero_()
    if k.grad is not None:
        k.grad.zero_()
    if v.grad is not None:
        v.grad.zero_()

    o, a = attn(q, k, v)
    L_o = o.diag().sum()
    L_o.backward()
    return q, k, v


def print_grads(q, k, v):
    print("dq:", q.grad, "\ndk:", k.grad, "\ndv:", v.grad, "\n------------")


def main():
    # test_sum_diag()
    # test_attn_grads()
    # test_pyattn_op()
    # test_my_attn_op()
    # test_Exp_op()
    test_nat_attn()


if __name__ == "__main__":
    main()
