from .blender import BlenderDataset
from .llff import LLFFDataset
from .phototourism import PhototourismDataset, NotreDameDataset

dataset_dict = {'blender': BlenderDataset,
                'llff': LLFFDataset,
                'phototourism': PhototourismDataset,
                'notredame': NotreDameDataset}