import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiLayerPerceptron(nn.Module):
    def __init__(self,
                 size_D=768,
                 hidden_layer_multiplier=4,
                 size_N=196,
                 print_stats=0,
                 pretrained_features=None):
        super().__init__()

        self.print_stats = print_stats
        self.size_D = size_D
        self.size_N = size_N
        self.hidden_layer_multiplier = hidden_layer_multiplier
        self.hidden_layer_size = self.hidden_layer_multiplier*self.size_D

        self.fc1 = nn.Linear(in_features=self.size_D,
                        out_features=self.hidden_layer_size)

        self.connection = nn.GELU()

        self.fc2 = nn.Linear(in_features=self.hidden_layer_size,
                        out_features=self.size_D)

        # TODO: Checkout the dropout and documentation for GELU + LayerNorm

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

    # Print optimizer's state_dict
    # print("Optimizer's state_dict:")
    # for var_name in optimizer.state_dict():
    #     print(var_name, "\t", optimizer.state_dict()[var_name])


if __name__ == "__main__":
    print_model_state_dict()