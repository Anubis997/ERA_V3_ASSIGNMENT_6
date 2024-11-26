import torch
from Train import MNIST_CNN

def test_model():
    model = MNIST_CNN()
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert param_count < 20000, f"Total parameter count is {param_count}, which exceeds the limit."

    # Check for Batch Normalization
    assert any(isinstance(layer, torch.nn.BatchNorm2d) for layer in model.features), "Batch Normalization not used."

    # Check for Dropout
    assert any(isinstance(layer, torch.nn.Dropout2d) for layer in model.features), "Dropout not used."

    # Check for either Fully Connected Layer or GAP
    has_fc = any(isinstance(layer, torch.nn.Linear) for layer in model.classifier)
    has_gap = any(isinstance(layer, torch.nn.AdaptiveAvgPool2d) for layer in model.classifier)
    
    assert has_fc or has_gap, "Neither Fully Connected Layer nor GAP is used."

if __name__ == "__main__":
    test_model()