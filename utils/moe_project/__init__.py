"""Small modular mixture-of-experts project. See README.md.

The models can be imported straight from the package, like the prebuilt Keras models:
    from utils.moe_project import CNNClassifier, MoEClassifier, PMoEClassifier, MGEClassifier, compile_default
"""
from .models import CNNClassifier, MoEClassifier, PMoEClassifier, MGEClassifier, compile_default
