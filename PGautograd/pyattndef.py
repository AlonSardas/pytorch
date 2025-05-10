import torch
from torch.autograd import Function


class MyAttnOp(Function):

    @staticmethod
    def forward(ctx, q, k, v):
        # print(input)
        # q, k, v = input
        print("inside forward")
        x = torch.matmul(q, k.transpose(0, 1))
        ctx.save_for_backward(q, k, v, x)
        a = torch.tanh(x)
        o = torch.matmul(a, v)
        return o, a
        # return o
        # return q.exp()

    @staticmethod
    def backward(ctx, grad1, grad2):
        print("my backward was called")
        q, k, v, x = ctx.saved_tensors
        print(grad1)
        q_grad = ((grad1 @ v.T) / (x.cosh()**2)) @ k
        return q_grad, None, None
    

class Exp2(Function):
    @staticmethod
    def forward(ctx, i):
        result = i.exp()
        ctx.save_for_backward(result)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        result, = ctx.saved_tensors
        return grad_output * result
