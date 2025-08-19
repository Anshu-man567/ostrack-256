import torch.nn as nn

"""Implementation of the MLP for the Transformer

This class implements the Mlp layer to be used as part of the Multi-Head Self Attention module

This module consists of the following layers:
1. Linear Projection for D -> 4*D
2. GELU
3. Linear Projection for 4*D -> D
"""
class MultiLayerPerceptron(nn.Module):

    """Initialize the module
        
        Args:
            size_D: Dimension of the input embeddings
            hidden_layer_multiplier: Multiplier for hidden layer size in MLP; 
                                     Multiplied to #size_D to use as the output size of the first layer
            size_N: number of input tokens
            print_stats: Enable printing of debug stats
    """
    def __init__(self,
                 size_D=768,
                 hidden_layer_multiplier=4,
                 size_N=196,
                 print_stats=0):

        super().__init__()

        self.print_stats = print_stats
        self.size_N = size_N
        self.size_D = size_D
        self.hidden_layer_multiplier = hidden_layer_multiplier
        self.hidden_layer_size = self.size_D * self.hidden_layer_multiplier

        self.fc1 = nn.Linear(in_features=self.size_D,
                             out_features=self.hidden_layer_size)

        self.connection = nn.GELU()

        self.fc2 = nn.Linear(in_features=self.hidden_layer_size,
                             out_features=self.size_D)

        # TODO (Anshu-man567) : Checkout the dropout and documentation for GELU + LayerNorm

    def forward(self, x):

        l1_output = self.fc1(x)
        if self.print_stats:
            print("size of output aft layer1", l1_output.size())

        l1a_output = self.connection(l1_output)
        if self.print_stats:
            print("size of output aft layer1 and GELU", l1a_output.size())

        l2_output = self.fc2(l1a_output)
        if self.print_stats:
            print("size of output aft layer2", l2_output.size())

        return l2_output

def print_model_state_dict():
    model = MultiLayerPerceptron()

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())


if __name__ == "__main__":
    print_model_state_dict()