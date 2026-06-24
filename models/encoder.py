import torch
import torch.nn as nn

# MLP-2L code
class TwoLayerMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(TwoLayerMLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# if __name__ == "__main__":
#     INPUT_FEATURES = 10
#     HIDDEN_NODES = 32
#     NUM_CLASSES = 2 

#     model = TwoLayerMLP(INPUT_FEATURES, HIDDEN_NODES, NUM_CLASSES)
#     dummy_input = torch.randn(5, INPUT_FEATURES)
#     predictions = model(dummy_input)
    
#     print("Output shape:", predictions.shape)
#     print("Predictions tensor:\n", predictions)