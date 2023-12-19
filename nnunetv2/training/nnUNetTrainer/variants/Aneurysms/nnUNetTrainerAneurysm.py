import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

class nnUNetTrainerAneurysm(nnUNetTrainer):
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict, unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        
        #Este parámetro define directamente el número de veces que la red recorrerá todo el conjunto de datos de entrenamiento.
        self.num_epochs = 100 #Por defecto 1000
        
        # Este parámetro define el número de iteraciones de entrenamiento por época. 
        # Sin embargo, el tamaño del lote real se calcula indirectamente dividiendo este número por el número de muestras de entrenamiento 
        # en su conjunto de datos. Por ejemplo, si tiene 10.000 muestras de entrenamiento, el tamaño del lote sería 10.000 / 250 = 40.
        self.num_iterations_per_epoch = 250 #Por defecto 250
        
        # Este parámetro funciona de manera similar al anterior pero determina el número de iteraciones de validación por época, 
        # definiendo indirectamente el tamaño del lote de validación
        self.num_val_iterations_per_epoch = 50 #Por defecto 50
        
        