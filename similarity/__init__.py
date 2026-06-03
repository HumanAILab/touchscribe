# Import specific modules to make them accessible via `similarity`
from .VGG16_similarity_Xing import get_frame_similarity

# Optional: Define __all__ to control what gets imported with `from similarity import *`
__all__ = [
    "get_frame_similarity",
]
