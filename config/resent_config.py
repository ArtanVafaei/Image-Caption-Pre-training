from dataclasses import dataclass

@dataclass
class resnet_config:
    num_layers = 2
    dropout = 0.1
    num_channels = 3
    output_embed = 64 * (2 ** num_layers)
    adapt_pool = 1024       # Make sure that the sqrt root of this number is a whole number. For example: The sqrt(1024) is 32, which is a whole number. 
