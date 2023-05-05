import torchvision.models as models

def get_mobilenet_v3_large(pretrained=True):
    return models.mobilenet_v3_large(pretrained=pretrained)

def get_mobilenet_v3_small(pretrained=True):
    return models.mobilenet_v3_small(pretrained=pretrained)
