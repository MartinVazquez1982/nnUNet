
import torch
import numpy as np

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from typing import List, Union, Tuple
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.utility_transforms import NumpyToTensor, RenameTransform, RemoveLabelTransform
from nnunetv2.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import \
    DownsampleSegForDSTransform2
from batchgenerators.transforms.spatial_transforms import SpatialTransform
<<<<<<< HEAD

#----- IMPORTS NECESARIOS PARA LAS FUNCIONES DE TRAINER DA5 -----
from batchgenerators.transforms.color_transforms import BrightnessTransform, ContrastAugmentationTransform, \
    GammaTransform
from batchgenerators.transforms.local_transforms import BrightnessGradientAdditiveTransform, LocalGammaTransform
from batchgenerators.transforms.noise_transforms import MedianFilterTransform, GaussianBlurTransform, \
    GaussianNoiseTransform, BlankRectangleTransform, SharpeningTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import SpatialTransform, Rot90Transform, TransposeAxesTransform, \
    MirrorTransform
from batchgenerators.transforms.utility_transforms import OneOfTransform

from nnunetv2.configuration import ANISO_THRESHOLD
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.data_augmentation.custom_transforms.cascade_transforms import MoveSegAsOneHotToData, \
    ApplyRandomBinaryOperatorTransform, RemoveRandomConnectedComponentFromOneHotEncodingTransform
from nnunetv2.training.data_augmentation.custom_transforms.deep_supervision_donwsampling import \
    DownsampleSegForDSTransform2
from nnunetv2.training.data_augmentation.custom_transforms.masking import MaskTransform
from nnunetv2.training.data_augmentation.custom_transforms.region_based_training import \
    ConvertSegmentationToRegionsTransform
from nnunetv2.training.data_augmentation.custom_transforms.transforms_for_dummy_2d import Convert3DTo2DTransform, \
    Convert2DTo3DTransform
=======
>>>>>>> 215834141f80a337a8db9733939e327837eb09aa

class nnUNetTrainerAneurysm(nnUNetTrainer):
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
         # -------------- NUMERO DE EPOCAS -------------------- #
        
        #Este parámetro define directamente el número de veces que la red recorrerá todo el conjunto de datos de entrenamiento.
<<<<<<< HEAD
        self.num_epochs = 400 #Por defecto 1000
=======
        self.num_epochs = 200 #Por defecto 1000
>>>>>>> 215834141f80a337a8db9733939e327837eb09aa
        
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
        
        # Este parametro es util para ajustar la atención del modelo durante el entrenamiento hacia los píxeles que corresponden al aneurisma. 
        # Si tu aneurisma es pequeño en comparación con el fondo en cada imagen, incrementar este parámetro podría ayudar al modelo a centrarse 
        # más en aprender las características específicas del aneurisma y mejorar su capacidad para segmentarlo con precisión.
        self.oversample_foreground_percent = 0.50 #Por defecto 0.33
    
<<<<<<< HEAD
    # @staticmethod
    # def get_training_transforms(patch_size: Union[np.ndarray, Tuple[int]],
    #                             rotation_for_DA: dict,
    #                             deep_supervision_scales: Union[List, Tuple, None],
    #                             mirror_axes: Tuple[int, ...],
    #                             do_dummy_2d_data_aug: bool,
    #                             order_resampling_data: int = 3,
    #                             order_resampling_seg: int = 1,
    #                             border_val_seg: int = -1,
    #                             use_mask_for_norm: List[bool] = None,
    #                             is_cascaded: bool = False,
    #                             foreground_labels: Union[Tuple[int, ...], List[int]] = None,
    #                             regions: List[Union[List[int], Tuple[int, ...], int]] = None,
    #                             ignore_label: int = None) -> AbstractTransform:
        
    #     tr_transforms = []       
      
    #     tr_transforms.append(SpatialTransform(
    #         patch_size, patch_center_dist_from_border=None,
    #         do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
    #         do_rotation=True, angle_x=rotation_for_DA['x'], angle_y=rotation_for_DA['y'], angle_z=rotation_for_DA['z'],
    #         p_rot_per_axis=1,  # todo experiment with this
    #         do_scale=True, scale=(0.7, 1.4),
    #         border_mode_data="constant", border_cval_data=0, order_data=order_resampling_data,
    #         border_mode_seg="constant", border_cval_seg=border_val_seg, order_seg=order_resampling_seg,
    #         random_crop=False,  # random cropping is part of our dataloaders
    #         p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
    #         independent_scale_for_each_axis=False  # todo experiment with this
    #     ))
        
    #     tr_transforms.append(RemoveLabelTransform(-1, 0))

    #     tr_transforms.append(RenameTransform('seg', 'target', True))

    #     if deep_supervision_scales is not None:
    #         tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
    #                                                           output_key='target'))
            
    #     tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        
    #     return Compose(tr_transforms)

    def configure_rotation_dummyDA_mirroring_and_inital_patch_size(self):
        """
        This function is stupid and certainly one of the weakest spots of this implementation. Not entirely sure how we can fix it.
        """
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)
        # todo rotation should be defined dynamically based on patch size (more isotropic patch sizes = more rotation)
        if dim == 2:
            do_dummy_2d_data_aug = False
            # todo revisit this parametrization
            if max(patch_size) / min(patch_size) > 1.5:
                rotation_for_DA = {
                    'x': (-15. / 360 * 2. * np.pi, 15. / 360 * 2. * np.pi),
                    'y': (0, 0),
                    'z': (0, 0)
                }
            else:
                rotation_for_DA = {
                    'x': (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi),
                    'y': (0, 0),
                    'z': (0, 0)
                }
            mirror_axes = (0, 1)
        elif dim == 3:
            # todo this is not ideal. We could also have patch_size (64, 16, 128) in which case a full 180deg 2d rot would be bad
            # order of the axes is determined by spacing, not image size
            do_dummy_2d_data_aug = (max(patch_size) / patch_size[0]) > ANISO_THRESHOLD
            if do_dummy_2d_data_aug:
                # why do we rotate 180 deg here all the time? We should also restrict it
                rotation_for_DA = {
                    'x': (-180. / 360 * 2. * np.pi, 180. / 360 * 2. * np.pi),
                    'y': (0, 0),
                    'z': (0, 0)
                }
            else:
                rotation_for_DA = {
                    'x': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    'y': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                    'z': (-30. / 360 * 2. * np.pi, 30. / 360 * 2. * np.pi),
                }
            mirror_axes = (0, 1, 2)
        else:
            raise RuntimeError()

        # todo this function is stupid. It doesn't even use the correct scale range (we keep things as they were in the
        #  old nnunet for now)
        initial_patch_size = get_patch_size(patch_size[-dim:],
                                            *rotation_for_DA.values(),
                                            (0.7, 1.43))
        if do_dummy_2d_data_aug:
            initial_patch_size[0] = patch_size[0]

        self.print_to_log_file(f'do_dummy_2d_data_aug: {do_dummy_2d_data_aug}')
        self.inference_allowed_mirroring_axes = mirror_axes

        return rotation_for_DA, do_dummy_2d_data_aug, initial_patch_size, mirror_axes

=======
>>>>>>> 215834141f80a337a8db9733939e327837eb09aa
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
<<<<<<< HEAD
        matching_axes = np.array([sum([i == j for j in patch_size]) for i in patch_size])
        valid_axes = list(np.where(matching_axes == np.max(matching_axes))[0])

        tr_transforms = []

        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            tr_transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None

        tr_transforms.append(
            SpatialTransform(
                patch_size_spatial,
                patch_center_dist_from_border=None,
                do_elastic_deform=False,
                do_rotation=True,
                angle_x=rotation_for_DA['x'],
                angle_y=rotation_for_DA['y'],
                angle_z=rotation_for_DA['z'],
                p_rot_per_axis=0.5,
                do_scale=True,
                scale=(0.7, 1.43),
                border_mode_data="constant",
                border_cval_data=0,
                order_data=order_resampling_data,
                border_mode_seg="constant",
                border_cval_seg=-1,
                order_seg=order_resampling_seg,
                random_crop=False,
                p_el_per_sample=0.2,
                p_scale_per_sample=0.2,
                p_rot_per_sample=0.4,
                independent_scale_for_each_axis=True,
            )
        )

        if do_dummy_2d_data_aug:
            tr_transforms.append(Convert2DTo3DTransform())

        if np.any(matching_axes > 1):
            tr_transforms.append(
                Rot90Transform(
                    (0, 1, 2, 3), axes=valid_axes, data_key='data', label_key='seg', p_per_sample=0.5
                ),
            )

        if np.any(matching_axes > 1):
            tr_transforms.append(
                TransposeAxesTransform(valid_axes, data_key='data', label_key='seg', p_per_sample=0.5)
            )

        tr_transforms.append(OneOfTransform([
            MedianFilterTransform(
                (2, 8),
                same_for_each_channel=False,
                p_per_sample=0.2,
                p_per_channel=0.5
            ),
            GaussianBlurTransform((0.3, 1.5),
                                  different_sigma_per_channel=True,
                                  p_per_sample=0.2,
                                  p_per_channel=0.5)
        ]))

        tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1))

        tr_transforms.append(BrightnessTransform(0,
                                                 0.5,
                                                 per_channel=True,
                                                 p_per_sample=0.1,
                                                 p_per_channel=0.5
                                                 )
                             )

        tr_transforms.append(OneOfTransform(
            [
                ContrastAugmentationTransform(
                    contrast_range=(0.5, 2),
                    preserve_range=True,
                    per_channel=True,
                    data_key='data',
                    p_per_sample=0.2,
                    p_per_channel=0.5
                ),
                ContrastAugmentationTransform(
                    contrast_range=(0.5, 2),
                    preserve_range=False,
                    per_channel=True,
                    data_key='data',
                    p_per_sample=0.2,
                    p_per_channel=0.5
                ),
            ]
        ))

        tr_transforms.append(
            SimulateLowResolutionTransform(zoom_range=(0.25, 1),
                                           per_channel=True,
                                           p_per_channel=0.5,
                                           order_downsample=0,
                                           order_upsample=3,
                                           p_per_sample=0.15,
                                           ignore_axes=ignore_axes
                                           )
        )

        tr_transforms.append(
            GammaTransform((0.7, 1.5), invert_image=True, per_channel=True, retain_stats=True, p_per_sample=0.1))
        tr_transforms.append(
            GammaTransform((0.7, 1.5), invert_image=True, per_channel=True, retain_stats=True, p_per_sample=0.1))

        if mirror_axes is not None and len(mirror_axes) > 0:
            tr_transforms.append(MirrorTransform(mirror_axes))

        tr_transforms.append(
            BlankRectangleTransform([[max(1, p // 10), p // 3] for p in patch_size],
                                    rectangle_value=np.mean,
                                    num_rectangles=(1, 5),
                                    force_square=False,
                                    p_per_sample=0.4,
                                    p_per_channel=0.5
                                    )
        )

        tr_transforms.append(
            BrightnessGradientAdditiveTransform(
                _brightnessadditive_localgamma_transform_scale,
                (-0.5, 1.5),
                max_strength=_brightness_gradient_additive_max_strength,
                mean_centered=False,
                same_for_all_channels=False,
                p_per_sample=0.3,
                p_per_channel=0.5
            )
        )

        tr_transforms.append(
            LocalGammaTransform(
                _brightnessadditive_localgamma_transform_scale,
                (-0.5, 1.5),
                _local_gamma_gamma,
                same_for_all_channels=False,
                p_per_sample=0.3,
                p_per_channel=0.5
            )
        )

        tr_transforms.append(
            SharpeningTransform(
                strength=(0.1, 1),
                same_for_each_channel=False,
                p_per_sample=0.2,
                p_per_channel=0.5
            )
        )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            tr_transforms.append(MaskTransform([i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                                               mask_idx_in_seg=0, set_outside_to=0))

        tr_transforms.append(RemoveLabelTransform(-1, 0))

        if is_cascaded:
            if ignore_label is not None:
                raise NotImplementedError('ignore label not yet supported in cascade')
            assert foreground_labels is not None, 'We need all_labels for cascade augmentations'
            use_labels = [i for i in foreground_labels if i != 0]
            tr_transforms.append(MoveSegAsOneHotToData(1, use_labels, 'seg', 'data'))
            tr_transforms.append(ApplyRandomBinaryOperatorTransform(
                channel_idx=list(range(-len(use_labels), 0)),
                p_per_sample=0.4,
                key="data",
                strel_size=(1, 8),
                p_per_label=1))
            tr_transforms.append(
                RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                    channel_idx=list(range(-len(use_labels), 0)),
                    key="data",
                    p_per_sample=0.2,
                    fill_with_other_class_p=0,
                    dont_do_if_covers_more_than_x_percent=0.15))

        tr_transforms.append(RenameTransform('seg', 'target', True))

        if regions is not None:
            # the ignore label must also be converted
            tr_transforms.append(ConvertSegmentationToRegionsTransform(list(regions) + [ignore_label]
                                                                       if ignore_label is not None else regions,
                                                                       'target', 'target'))

        if deep_supervision_scales is not None:
            tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
                                                              output_key='target'))
        tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
        tr_transforms = Compose(tr_transforms)
        return tr_transforms
    
def _brightnessadditive_localgamma_transform_scale(x, y):
    return np.exp(np.random.uniform(np.log(x[y] // 6), np.log(x[y])))

def _brightness_gradient_additive_max_strength(_x, _y):
    return np.random.uniform(-5, -1) if np.random.uniform() < 0.5 else np.random.uniform(1, 5)

def _local_gamma_gamma():
    return np.random.uniform(0.01, 0.8) if np.random.uniform() < 0.5 else np.random.uniform(1.5, 4)
=======
        
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
>>>>>>> 215834141f80a337a8db9733939e327837eb09aa
