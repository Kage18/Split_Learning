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
from model import ClientNN, ServerNN
from plotting import generate_simple_plot


def parse_args():

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
    model = ServerNN()
    model = model.to(device)

    # Define the loss criterion
    loss_crit = nn.CrossEntropyLoss()

    # Use Stochastic Gradient Descent with momentum and weight decay
    # as the optimizer
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate,
            momentum=0.9, weight_decay=5e-4)

    total_training_time = 0.0

    epoch, step, batch_idx = 1, 0, 0

    curr, phase = 1, "train"

    val_loss, val_losses, val_accs = 0.0, [], []
    total_n_labels_train, total_n_labels_test = 0, 0
    correct_train, correct_test = 0, 0

    while(True):
        msg = comm.recv(source=curr)

        if msg[0] == "tensor_and_labels":
            if phase == "train":
                optimizer.zero_grad()

            input_tensor, labels = msg[1]

            logits = model(input_tensor)

            _, predictions = logits.max(1)

            loss = loss_crit(logits, labels)

            if phase == "train":
                total_n_labels_train += len(labels)

                correct_train += predictions.eq(labels).sum().item()

                loss.backward()

                optimizer.step()

                comm.send(input_tensor.grad, dest=curr)

                batch_idx += 1

                if batch_idx % args.log_steps == 0:
                    acc = correct_train / total_n_labels_train

                    print('{} - Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}'.format(
                        curr, epoch, int((
                        args.num_batches * args.batch_size) / max_rank * (
                        curr-1)) + batch_idx * args.batch_size,
                        args.num_batches * args.batch_size, 100. * (((
                        args.num_batches / max_rank * (curr-1)) + \
                        batch_idx) / args.num_batches), loss.item()))

            if phase == "test":
                step += 1
                total_n_labels_test += len(labels)

                correct_test += predictions.eq(labels).sum().item()
                val_loss += loss.item()

        elif msg[0] == "time":
            total_training_time += msg[1]

        elif msg == "worker_done":
            if curr == max_rank:
                epoch += 1

            curr = (curr % max_rank) + 1
            phase = "train"

            total_n_labels_train, correct_train, batch_idx = 0, 0 ,0

        elif msg == "epoch_done" or msg == "training_complete":
            val_loss /= step
            val_losses.append(val_loss)

            acc = correct_test / total_n_labels_test
            val_accs.append(acc)

            print("\nTest set - Epoch: {} - Loss: {:.4f}, Acc: ({:2f}%)\n".format(
                epoch, val_loss, 100 * acc))

            if curr == max_rank:
                epoch += 1

            curr = (curr % max_rank) + 1
            phase = "train"

            total_n_labels_test, correct_test = 0, 0
            step, batch_idx = 0, 0

            if msg == "training_complete":
                print("Training complete.")

                epoch_list = list(range(1, args.epochs+1))
                generate_simple_plot(epoch_list, val_losses,
                        "Test loss (Split Learning)", "epoch", "loss", [0.3, 0.9],
                        save=True, fname="test_loss_sl.pdf")
                generate_simple_plot(epoch_list, val_accs,
                        "Test accuracy (Split Learning)", "epoch", "accuracy",
                        [0.65, 1.0], save=True, fname="test_acc_sl.pdf")

                print("Total training time: {:.2f}s".format(total_training_time))
                print("Final test accuracy: {:.4f}".format(acc))
                print("Final test loss: {:.4f}".format(val_loss))

                exit()

            # Only reset test loss if training not complete
            val_loss = 0.0

        elif msg == "test":
            phase = "test"
            step , total_n_labels_train, correct_train = 0, 0, 0

        elif msg == "data_downloaded":
            exit()



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

        if msg == "Comienzo":
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
                comm.send("test", dest=SERVER)

                for batch_idx, (inputs, labels) in enumerate(test_loader):
                    
                    inputs, labels = inputs.to(device), labels.to(device)

                    split_layer_tensor = model(inputs)

                    comm.send(["tensor_and_labels", [split_layer_tensor, labels]],
                        dest=SERVER)

                del split_layer_tensor, inputs, labels
                torch.cuda.empty_cache()

            comm.send("Comienzo", dest=onright)

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

