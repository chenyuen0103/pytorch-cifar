import os
import importlib
from models import *  # Import custom models from the models directory
from torchvision import models as torchvision_models

def get_model(model_name, num_classes, pretrained=False):
    """
    Dynamically load a model based on its name.
    Searches both custom models and torchvision models.
    
    Args:
        model_name (str): Name of the model to load.
        num_classes (int): Number of output classes for the model.
        pretrained (bool): Whether to load pretrained weights (only for torchvision models).

    Returns:
        torch.nn.Module: The initialized model.
    """
    # Check if the model is in the custom models directory
    if model_name in globals():
        # Instantiate the custom model
        return globals()[model_name](num_classes=num_classes)
    
    # Check if the model is in torchvision
    elif hasattr(torchvision_models, model_name):
        # Instantiate the torchvision model
        model_cls = getattr(torchvision_models, model_name)
        return model_cls(pretrained=pretrained, num_classes=num_classes)
    
    else:
        raise ValueError(f"Model '{model_name}' not found in custom models or torchvision models.")
