from copy import deepcopy
from tqdm import tqdm

import torch
import torchvision.datasets
import torchvision.models
import torchvision.transforms
import numpy as np

from classification_rbm import ClassificationRBM
from util import train_rbm, test_rbm_model
np.random.seed(3)
torch.manual_seed(3) # Keeping it fixed

import argparse

parser = argparse.ArgumentParser(description='classification_model text classificer')
# learning
parser.add_argument('--lr', type=float, default=0.05, help='initial learning rate [default: 0.001]')
parser.add_argument('--epochs', type=int, default=50, help='number of epochs for train [default: 256]')
parser.add_argument('--batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('--early-stop', type=int, default=15, help='iteration numbers to stop without performance increasing')
parser.add_argument('--visible-units', type=int, default=784, help='Number of dimensions in input')
parser.add_argument('--hidden-units', type=int, default = 500, help = 'Number of dimensions of the hidden representation')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disable the gpu')
parser.add_argument('--cd-k', type = int, default = 1, help = 'The K in the contrastive Divergence Algorithm')
parser.add_argument('--type', type = str, default = 'discriminative', help = 'The type of training you want to start - discriminative, hybrid and generative' )
parser.add_argument('--sparsity-coeffectient', type = float, default = 0.00, help = 'The amount that must be subtracted from bias after every update')
parser.add_argument('--data-folder', type = str, default = 'data/mnist', help = 'Folder in which the data needs to be stored')
parser.add_argument('--generative-factor', type = int, default = 0.01)

args = parser.parse_args()


DATA_FOLDER = 'data/mnist'

args.cuda = torch.cuda.is_available() and not args.no_cuda
CUDA_DEVICE = 0

if args.cuda:
    torch.cuda.set_device(CUDA_DEVICE)
    




########## LOADING DATASET ##########
print('Loading dataset...')

train_dataset = torchvision.datasets.MNIST(root=args.data_folder, train=True, transform=torchvision.transforms.ToTensor(), download=True)

train_dataset, validation_dataset = torch.utils.data.random_split(train_dataset, (50000, 10000))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=args.batch_size)



test_dataset = torchvision.datasets.MNIST(root=args.data_folder, train=False, transform=torchvision.transforms.ToTensor(), download=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size)


########## TRAINING RBM ##########

rbm = ClassificationRBM(args.visible_units, args.hidden_units, args.cd_k,  learning_rate = args.lr, use_cuda=args.cuda)


loss_list, best_model = train_rbm(rbm, train_loader, args, validation_loader, args.type, args.generative_factor)


######### TESTING RBM ##########

test_rbm_model(best_model, test_loader, args)