import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainerAneurysm(nnUNetTrainer):
    
    def _init_(self):
        sinError=None