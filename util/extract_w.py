# from tabnanny import check
import torch
import pathlib


model_path = pathlib.Path("G_first.pth")

assert model_path.exists(), "model path does not exist"

checkpoint = torch.load(str(model_path), map_location='cpu')

state_file = checkpoint['model']
iteration = checkpoint['iteration']

out_path = model_path.parent / pathlib.Path("19_"+str(iteration)+'_demo'+'.pth')
out_path = str(out_path)
torch.save({'model': state_file,
            'iteration': iteration,
            'optimizer': None,
            'learning_rate': None}, out_path)
