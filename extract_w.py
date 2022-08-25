from tabnanny import check
import torch

file_path = ""
out_path = file_path[:-4]+'_demo'+file_path[-4:]

# with open (file_path,'r',encoding='uft-8') as f1:
#     with open (,'w',encoding=  'uft-8') as f2:
        # model = torch.load(file)
checkpoint = torch.load(file_path,map_location='cpu')

state_file = checkpoint['model']
iteration = checkpoint['iteration']


torch.save({'model': state_file,
              'iteration': iteration,
              'optimizer': None,
              'learning_rate': None}, out_path)


