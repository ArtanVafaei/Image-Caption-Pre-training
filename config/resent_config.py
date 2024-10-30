from dataclasses import dataclass

@dataclass
class resnet_config:
    num_layers = 3
    dropout = 0.1
    num_channels = 3         # DO NOT CHANGE: We are using RGB Images, which have 3 channels.
    output_embed = 64 * (2 ** num_layers)          # DO NOT CHANGE  
    adapt_pool = 1024       # Make sure that the sqrt root of this number is a whole number. For example: The sqrt(1024) is 32, which is a whole number. 
