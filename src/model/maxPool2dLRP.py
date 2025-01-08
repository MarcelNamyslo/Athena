import torch
import torch.nn.functional as F
from torch.autograd import Function
from rules.maxPoolRules import lrp_maxpool_rule
from .cache import set_relevance_cache
class MaxPool2dLRP(Function):
    @staticmethod
    def forward(ctx, input, kernel_size=2, stride=None, padding=0):
        """
        Standard forward pass for max-pooling with saving indices for backward relevance propagation.
        """
        ctx.save_for_backward(input)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding
        output, indices = F.max_pool2d(input, kernel_size=kernel_size, stride=stride, padding=padding, return_indices=True)
        ctx.indices = indices

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Custom backward pass that first performs the standard gradient computation and then calculates relevance.
        """
        input, = ctx.saved_tensors
        indices = ctx.indices

        grad_input = F.max_unpool2d(
            grad_output,  
            indices,     
            output_size=input.shape, 
            kernel_size=ctx.kernel_size,
            stride=ctx.stride,
            padding=ctx.padding
        )
        relevance_input = relevance_input = lrp_maxpool_rule(input, grad_output, kernel_size=ctx.kernel_size, stride=ctx.stride, padding=ctx.padding)
        set_relevance_cache(relevance_input) 
        return grad_input, None, None, None

class MaxPool2dLRPLayer(torch.nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(MaxPool2dLRPLayer, self).__init__()
        self.maxpool = torch.nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x, explain=False, lrp= False, rule="epsilon", **kwargs):
        if explain or lrp:
            return MaxPool2dLRP.apply(x, self.maxpool.kernel_size, self.maxpool.stride, self.maxpool.padding)
        return self.maxpool(x)
