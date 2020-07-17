from mpi4py import MPI

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

import os
from time import time
import pickle
import itertools
import numpy as np
from sys import argv
from argparse import ArgumentParser, Namespace

from models import ClientNN, ServerNN
from plotting import generate_simple_plot


def parse_args() -> Namespace:
    """Parses CL arguments

    Returns:
        Namespace object containing all arguments
    """
    parser = ArgumentParser()

    parser.add_argument("-bs", "--batch_size", type=int, default=64)
    parser.add_argument("-nb", "--num_batches", type=int, default=938)
    parser.add_argument("-tbs", "--test_batch_size", type=int, default=1000)
    parser.add_argument("-ls", "--log_steps", type=int, default=50)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.01)
    parser.add_argument("-e", "--epochs", type=int, default=10)
    parser.add_argument("-p", "--plot", type=bool, default=True)

    return parser.parse_args(argv[1:])

args = parse_args()


torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#communicator
comm = MPI.COMM_WORLD
max_rank = comm.Get_size() - 1
rank = comm.Get_rank()

SERVER = 0


#Server
if rank == 0:
    continue

#Client
else:

    onleft = rank - 1
    onright = rank + 1
    curr = rank
    epoch = 1

    if curr == 1:
        onleft = max_rank
    
    if curr == max_rank:
        onright = 1
        comm.send("Comienzo", dest=onright)


    data_transformer = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])
    file_path = "./data/FashionMNIST/processed/fashion_mnist.pkl"
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as f:
            train_set = pickle.load(f)
    else:
        if rank == 1:
            train_set = torchvision.datasets.FashionMNIST(root='./data',
                    train=True, download=True, transform=data_transformer)
            train_loader = torch.utils.data.DataLoader(train_set,
                    batch_size=args.batch_size, shuffle=True)

            train_set = []
            for i, l in train_loader:
                train_set.append((i, l))

            with open(file_path, 'wb') as f:
                pickle.dump(train_set, f, protocol=pickle.HIGHEST_PROTOCOL)

            print("Data downloaded. Please run the script again.")
            comm.send("data_downloaded", dest=SERVER)
        exit()

    start = int(np.floor(len(train_set)/max_rank*(rank-1)))
    stop = int(np.floor(len(train_set)/max_rank*(rank)))
    train_set = train_set[start:stop]


    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
        download=True, transform=data_transformer)
    test_loader = torch.utils.data.DataLoader(testset,
            batch_size=args.test_batch_size, shuffle=False)


    model = ClientNN().to(device)

    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
            momentum=0.9, weight_decay=5e-4)

    while True:

        msg = comm.recv(source=onleft)

        if msg == "you_can_start":
            if rank == 1:
                print(f"\nStart epoch {epoch}:")

            start = time()
            for batch_idx, (inputs, labels) in enumerate(train_set):

                inputs, labels = inputs.to(device), labels.to(device)


                optimizer.zero_grad()


                split_layer_tensor = model(inputs)

                comm.send(["tensor_and_labels", [split_layer_tensor, labels]],
                    dest=SERVER)

                grads = comm.recv(source=SERVER)

                split_layer_tensor.backward(grads)

                optimizer.step()

            del split_layer_tensor, grads, inputs, labels
            torch.cuda.empty_cache()
            end = time()

            comm.send(["time", end-start], dest=SERVER)

            if rank == max_rank:
                comm.send("validation", dest=SERVER)

                for batch_idx, (inputs, labels) in enumerate(test_loader):
                    
                    inputs, labels = inputs.to(device), labels.to(device)

                    split_layer_tensor = model(inputs)

                    comm.send(["tensor_and_labels", [split_layer_tensor, labels]],
                        dest=SERVER)

                del split_layer_tensor, inputs, labels
                torch.cuda.empty_cache()

            comm.send("you_can_start", dest=onright)

            if epoch == args.epochs:
                msg="training_complete" if rank == max_rank else "worker_done"
                comm.send(msg, dest=SERVER)
                exit()
            else:
                msg="epoch_done" if rank == max_rank else "worker_done"
                comm.send(msg, dest=SERVER)

            epoch += 1


        

        data_transformer = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))])

