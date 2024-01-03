import torch
import numpy as np

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from typing import List, Union, Tuple
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor, RenameTransform, RemoveLabelTransform
from nnunetv2.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import \
    DownsampleSegForDSTransform2
from batchgenerators.transforms.spatial_transforms import SpatialTransform

class nnUNetTrainerAneurysm(nnUNetTrainer):
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
         # -------------- NUMERO DE EPOCAS -------------------- #
        
        #Este parámetro define directamente el número de veces que la red recorrerá todo el conjunto de datos de entrenamiento.
        self.num_epochs = 200 #Por defecto 1000
        
        # ---------- NUMERO DE ITERACIONES POR EPOCA --> TAMAÑO DEL LOTE -------------------------- #
        
        # Este parámetro define el número de iteraciones de entrenamiento por época. 
        # Sin embargo, el tamaño del lote real se calcula indirectamente dividiendo este número por el número de muestras de entrenamiento 
        # en su conjunto de datos. Por ejemplo, si tiene 10.000 muestras de entrenamiento, el tamaño del lote sería 10.000 / 250 = 40.
        # self.num_iterations_per_epoch = 250 #Por defecto 250
        
        # Segun chatgpt al ser un numero tan pequenio de muestras (15) lo mejor es un tamaño de lote pequeño tambien. Por eso con 5 iteraciones por epoca
        # quedaria 15/3, es decir un tamaño de lote de 5
        self.num_iterations_per_epoch = 3 #Por defecto 250
        
        # # Este parámetro funciona de manera similar al anterior pero determina el número de iteraciones de validación por época, 
        # # definiendo indirectamente el tamaño del lote de validación
        # self.num_val_iterations_per_epoch = 50 #Por defecto 50
    
    @staticmethod
    def get_training_transforms(patch_size: Union[np.ndarray, Tuple[int]],
                                rotation_for_DA: dict,
                                deep_supervision_scales: Union[List, Tuple, None],
                                mirror_axes: Tuple[int, ...],
                                do_dummy_2d_data_aug: bool,
                                order_resampling_data: int = 3,
                                order_resampling_seg: int = 1,
                                border_val_seg: int = -1,
                                use_mask_for_norm: List[bool] = None,
                                is_cascaded: bool = False,
                                foreground_labels: Union[Tuple[int, ...], List[int]] = None,
                                regions: List[Union[List[int], Tuple[int, ...], int]] = None,
                                ignore_label: int = None) -> AbstractTransform:
        
        tr_transforms = []       
        
        tr_transforms.append(SpatialTransform(
            patch_size, patch_center_dist_from_border=None,
            do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
            do_rotation=True, angle_x=rotation_for_DA['x'], angle_y=rotation_for_DA['y'], angle_z=rotation_for_DA['z'],
            p_rot_per_axis=1,  # todo experiment with this
            do_scale=True, scale=(0.7, 1.4),
            border_mode_data="constant", border_cval_data=0, order_data=order_resampling_data,
            border_mode_seg="constant", border_cval_seg=border_val_seg, order_seg=order_resampling_seg,
            random_crop=False,  # random cropping is part of our dataloaders
            p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
            independent_scale_for_each_axis=False  # todo experiment with this
        ))
        
        tr_transforms.append(RemoveLabelTransform(-1, 0))

        tr_transforms.append(RenameTransform('seg', 'target', True))

        if deep_supervision_scales is not None:
            tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                              output_key='target'))
            
        tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        
        return Compose(tr_transforms)