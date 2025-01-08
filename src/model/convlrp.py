import torch
import torch.nn.functional as F
from torch.autograd import Function
from rules.convRules import apply_lrp_rule
from .cache import set_relevance_cache, get_relevance_cache

class Conv2dLRP(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, rule='lrp0', explain = False, factor = 0.01, lrp = False, kwargs=None):
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.rule = rule
        ctx.explain = explain
        ctx.lrp = lrp
        ctx.factor = factor 
        ctx.kwargs = kwargs  

        return F.conv2d(input, weight, bias, stride, padding, dilation, groups)

    @staticmethod
    def backward(ctx, grad_output):
        relevance_cache = get_relevance_cache() 
        input, weight, bias = ctx.saved_tensors
      
        if ctx.lrp:
            if relevance_cache is None:
                raise ValueError("Relevance cache is empty. Something went wrong in the backward pass.")
            grad_output *= (1 +  ctx.factor * relevance_cache)  
            relevance_input = apply_lrp_rule(ctx.rule, input, weight, bias, ctx, grad_output, **ctx.kwargs)
            set_relevance_cache(relevance_input) 

            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation, groups=ctx.groups)
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride=ctx.stride, padding=ctx.padding, dilation=ctx.dilation, groups=ctx.groups)
            grad_bias = grad_output.sum(dim=(0, 2, 3)) if bias is not None else None
            return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None, None, None
        if ctx.explain:
            relevance_input = apply_lrp_rule(ctx.rule, input, weight, bias, ctx, grad_output, **ctx.kwargs)
            return relevance_input, None, None, None, None, None, None, None, None, None, None, None
    
        


class Conv2dLRPLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dLRPLayer, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=bias)

    def forward(self, x, explain=False, lrp = False, rule='lrp0', factor = 0.01, **kwargs):
        if lrp:
            return Conv2dLRP.apply(x, self.conv.weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups, rule, explain, factor, lrp, kwargs)
        if explain:
            return Conv2dLRP.apply(x, self.conv.weight, self.conv.bias, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups, rule, explain, factor, lrp, kwargs)

        return self.conv(x)
