from tabnanny import check
import torch

file_path = ""
out_path = file_path[:-4]+'_demo'+file_path[-4:]


checkpoint = torch.load(file_path, map_location='cpu')

state_file = checkpoint['model']
iteration = checkpoint['iteration']


torch.save({'model': state_file,
            'iteration': iteration,
            'optimizer': None,
            'learning_rate': None}, out_path)
