import torch
import torch.nn.functional as F

def lrp_maxpool_rule(input, relevance_output, kernel_size=2, stride=2, padding=0):
    """
    Define how relevance scores are propagated through max-pooling layers.
    
    Args:
        input: Input tensor to the max-pooling layer.
        relevance_output: Relevance scores of the output of the max-pooling layer.
        kernel_size: Size of the max-pooling kernel.
        stride: Stride used in max-pooling.
        padding: Padding used in max-pooling.

    Returns:
        relevance_input: Relevance scores for the input tensor of the max-pooling layer.
    """
    # Get the indices of the max values selected by max pooling
    _, indices = F.max_pool2d(input, kernel_size=kernel_size, stride=stride, padding=padding, return_indices=True)

    # Create a zero tensor for relevance propagation
    relevance_input = torch.zeros_like(input)

    flat_indices = indices.view(-1)

    # Flatten relevance_output to match the indices
    flat_relevance_output = relevance_output.view(-1)

    # Flatten relevance_input to ensure scatter operation can correctly add values
    flat_relevance_input = relevance_input.view(-1)

    # Propagate relevance from output to input based on indices
    flat_relevance_input.scatter_add_(0, flat_indices, flat_relevance_output)

    # Reshape relevance_input back to its original shape
    relevance_input = flat_relevance_input.view_as(input)

    return relevance_input
 