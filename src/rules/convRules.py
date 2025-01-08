import torch
import torch.nn.functional as F

def apply_lrp_rule(rule, input, weight, bias, ctx, relevance_output, **kwargs):
    if rule == 'lrp0':  # Standard LRP-0 rule
        return lrp_0(input, weight, bias, ctx, relevance_output)
    elif rule == 'epsilon':
        return lrp_epsilon(input, weight, bias, ctx, relevance_output, **kwargs)
    elif rule == 'gamma':
        return lrp_gamma(input, weight, bias, ctx, relevance_output, **kwargs)
    elif rule == 'alpha1beta0':
        return lrp_alpha1beta0(input, weight, bias, ctx, relevance_output, **kwargs)
    
    else:
        raise ValueError(f"Unknown LRP rule: {rule}")

# Implementation of basic LRP rule (LRP-0)
def lrp_0(input, weight, bias, ctx, relevance_output):
    
    Z = F.conv2d(input, weight, bias, ctx.stride, ctx.padding, ctx.dilation, ctx.groups) + 1e-9
    relevance_output = relevance_output / Z
    relevance_input = F.conv_transpose2d(relevance_output, weight, bias=None, stride=ctx.stride, padding=ctx.padding)
    return relevance_input * input
    

# Implementation of LRP epsilon rule
def lrp_epsilon(input, weight, bias, ctx, relevance_output, epsilon=1e-9):
    Z = F.conv2d(input, weight, bias, ctx.stride, ctx.padding, ctx.dilation, ctx.groups) + epsilon
    relevance_output = relevance_output / Z
    relevance_input = F.conv_transpose2d(relevance_output, weight, bias=None, stride=ctx.stride, padding=ctx.padding)
    return relevance_input * input


# Implementation of LRP gamma rule
def lrp_gamma(input, weight, bias, ctx, relevance_output, gamma=0.1):
   
    weight_pos = weight.clamp(min=0) * (1 + gamma)
    Z = F.conv2d(input, weight_pos, bias, ctx.stride, ctx.padding, ctx.dilation, ctx.groups) + 1e-9
    relevance_output = relevance_output / Z
    relevance_input = F.conv_transpose2d(relevance_output, weight_pos, bias=None, stride=ctx.stride, padding=ctx.padding)
    return relevance_input * input

# Implementation of LRP alpha1 beta 0
def lrp_alpha1beta0(input, weight, bias, ctx, relevance_output):
    weight_pos = weight.clamp(min=0)
    Z = F.conv2d(input, weight_pos, bias, ctx.stride, ctx.padding, ctx.dilation, ctx.groups) + 1e-9
    relevance_output = relevance_output / Z
    relevance_input = F.conv_transpose2d(relevance_output, weight_pos, bias=None, stride=ctx.stride, padding=ctx.padding)
    return relevance_input * input
