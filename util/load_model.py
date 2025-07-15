import logging

def load_yolo_model(model_name):
    from ultralytics import YOLO
    logging.getLogger('ultralytics').setLevel(logging.ERROR)

    return YOLO(model_name, verbose=False)

def load_reid_model(model_name):
    import torch
    import torchreid
    logging.getLogger('torchreid').setLevel(logging.ERROR)
    
    all_configs = get_reid_model_configs()
    if not model_name in all_configs:
        raise Exception('reid model is not configured')
    
    config = all_configs[model_name]

    # load model weights & load model structure
    checkpoint = torch.load(config.path, map_location='cpu')
    model = torchreid.models.build_model(name=model_name, num_classes=config.num_classes, pretrained=False)
    
    # move weights into structure
    state_dict = {}
    for key, value in checkpoint.items():
        if key.startswith('module.'):
            state_dict[key[7:]] = value  # Remove 'module.' prefix
        else:
            state_dict[key] = value
    
    model.load_state_dict(state_dict)

    model.eval()
    model.verbose = False
    return model

def get_reid_model_input_layer(model):
    first_layer = next(model.children())
    
    if hasattr(first_layer, 'in_channels'):
        return { 'layer': first_layer, 'input_channels': first_layer.in_channels, 'layer_type': type(first_layer).__name__ }
    else:
        return { 'layer': first_layer, 'layer_type': type(first_layer).__name__ }

def get_reid_model_configs():
    class Config:
        def __init__(self, path, num_classes):
            self.path = path
            self.num_classes = num_classes

    config = {
        'osnet_ain_x1_0': Config('models/osnet_ain_x1_0.pth', 4101),
        'osnet_x0_25': Config('models/osnet_x0_25.pth', 4101),
    }
    return config