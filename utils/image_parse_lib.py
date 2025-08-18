from enum import Enum


class ImageParseLib(Enum):
    # BEST FOR DEBUGGING
    PIL = 1
    # CAUSE PYTORCH has to be used
    TORCHVISION = 2
    # DO NOT USE for debugging at all
    OPENCV = 3
