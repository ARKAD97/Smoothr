from omegaconf import OmegaConf

from src.datatypes import Size2D

# to easily create objects in hydra config
OmegaConf.register_new_resolver("size2d", Size2D)
