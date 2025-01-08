import torch
import torch.nn.functional as F


def apply_lrp_rule(rule, input, weight, bias, ctx, relevance_output, **kwargs):
    if rule == 'epsilon':
        return lrp_epsilon(input, weight, bias, ctx, relevance_output, **kwargs)
    elif rule == 'gamma':
        return lrp_gamma(input, weight, bias, ctx, relevance_output, **kwargs)
    elif rule == 'alpha1beta0':
        return lrp_alpha1beta0(input, weight, bias, ctx, relevance_output, **kwargs)
    elif rule == 'lrp0':  # Standard LRP-0 rule
        return lrp_0(input, weight, bias, ctx, relevance_output)
    else:
        raise ValueError(f"Unknown LRP rule: {rule}")


def lrp_0(input, weight, bias, ctx, relevance_output):
    """
    Standard LRP-0 Rule Implementation.

    This function redistributes the relevance scores from the output neurons to the input neurons 
    proportionally based on their contribution to the output.

    Args:
        input (torch.Tensor): The input tensor to the layer (e.g., shape: [batch_size, in_features]).
        weight (torch.Tensor): The weight tensor of the layer (e.g., shape: [out_features, in_features]).
        bias (torch.Tensor): The bias tensor of the layer (if any).
        ctx (any): Context object (not used in this function).
        relevance_output (torch.Tensor): The relevance scores at the output of the layer.

    Returns:
        torch.Tensor: The relevance scores at the input of the layer.
    """
    Z = F.linear(input, weight, bias) + 1e-9  
    contributions = input.unsqueeze(-1) * weight.t()  
    proportion_contrib = contributions / Z.unsqueeze(1)  
    relevance_input = torch.matmul(proportion_contrib, relevance_output.unsqueeze(-1)) 
    relevance_input = relevance_input.squeeze(-1)  
    

    return relevance_input

# Implementation of LRP epsilon rule
def lrp_epsilon(input, weight, bias, ctx, relevance_output, epsilon=1e-9):
    Z = F.linear(input, weight, bias) + epsilon
    relevance_output = relevance_output / Z
    relevance_input = F.linear(relevance_output, weight.t(), bias=None)
    return relevance_input * input

# Implementation of LRP gamma rule
def lrp_gamma(input, weight, bias, ctx, relevance_output, gamma=0.1):
    weight_pos = weight.clamp(min=0) * (1 + gamma)
    Z = F.linear(input, weight_pos, bias) + 1e-9
    relevance_output = relevance_output / Z
    relevance_input = F.linear(relevance_output, weight_pos.t(), bias=None)
    return relevance_input * input

# Implementation of LRP alpha1 beta 0
def lrp_alpha1beta0(input, weight, bias, ctx, relevance_output):
    weight_pos = weight.clamp(min=0)
    Z = F.linear(input, weight_pos, bias) + 1e-9
    relevance_output = relevance_output / Z
    relevance_input = F.linear(relevance_output, weight_pos.t(), bias=None)
    return relevance_input * input
