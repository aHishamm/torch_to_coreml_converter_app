import torch 
import torchvision 
import coremltools as ct 
import argparse 
#Loading the EfficientSAM model
#Path to vits 
#/Users/ahishamm/Documents/projects/efficientSAM/EfficientSAM/torchscripted_model/efficient_sam_vits_torchscript.pt
#Path to vitt 
#/Users/ahishamm/Documents/projects/efficientSAM/EfficientSAM/torchscripted_model/efficient_sam_vitt_torchscript.pt

parser = argparse.ArgumentParser(description='Add the path to the Pytorch model: ')
parser.add_argument('--path',type=str,help='Specify the path to the Pytorch model',required=True) 
args = parser.parse_args() 
print(f'Path to Pytorch model: {args.path}')
#Loading the model
model = torch.jit.load(args.path) 
model.eval() 
