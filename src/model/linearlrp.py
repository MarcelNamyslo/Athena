
import torch
import torch.nn.functional as F
from torch.autograd import Function
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from rules.linearRules import apply_lrp_rule
from .cache import set_relevance_cache, get_relevance_cache

class LinearLRP(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, rule='lrp0', kwargs_dict={}, explain=False, lrp = False, is_output_layer=False, factor=1):

        ctx.save_for_backward(input, weight, bias)
        ctx.rule = rule
        ctx.kwargs = kwargs_dict
        ctx.explain = explain
        ctx.lrp = lrp
        ctx.factor = factor 
        ctx.is_output_layer = is_output_layer

        return F.linear(input, weight, bias)
    
    @staticmethod
    def backward(ctx, grad_output):
        return LinearLRP.backward_gradient(ctx, grad_output)


    @staticmethod
    def backward_gradient(ctx, grad_output):
        """
        Gradient-based backward method.
        """
        input, weight, bias = ctx.saved_tensors

        if ctx.lrp:
            if ctx.is_output_layer:
                relevance_input = apply_lrp_rule(ctx.rule, input, weight, bias, ctx, grad_output, **ctx.kwargs)
                set_relevance_cache(relevance_input)
            else:
                relevance_cache = get_relevance_cache()
                if relevance_cache is None:
                    raise ValueError("[ERROR] Relevance cache is empty. Something went wrong in the backward pass.")
                grad_output = grad_output * (1 + ctx.factor * relevance_cache)
                relevance_input = apply_lrp_rule(ctx.rule, input, weight, bias, ctx, grad_output, **ctx.kwargs)
                set_relevance_cache(relevance_input)

            grad_input = grad_output.mm(weight)
            grad_weight = grad_output.t().mm(input)
            grad_bias = grad_output.sum(0) if bias is not None else None

            return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None

        if ctx.explain:
            relevance_input = apply_lrp_rule(ctx.rule, input, weight, bias, ctx, grad_output, **ctx.kwargs)
            return relevance_input, None, None, None, None, None, None, None, None, None

        grad_input = grad_output @ weight
        grad_weight = grad_output.t() @ input
        grad_bias = grad_output.sum(0) if bias is not None else None

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None

class LinearLRPLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(LinearLRPLayer, self).__init__()
        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x, explain=False, lrp = False, rule='lrp0', is_output_layer=False, factor=1,**kwargs):
        if lrp:
            relevance_scores = LinearLRP.apply(x, self.linear.weight, self.linear.bias, rule, kwargs, explain, lrp,  is_output_layer, factor)
            return relevance_scores
        if explain:
            relevance_scores = LinearLRP.apply(x, self.linear.weight, self.linear.bias, rule, kwargs, explain, lrp, is_output_layer, factor)
            return relevance_scores
        else:
            return self.linear(x)
    
