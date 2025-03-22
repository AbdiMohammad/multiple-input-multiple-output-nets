#
# Copyright 2023- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#

# ==================================================================================================
# IMPORTS
# ==================================================================================================
import torch
from models.superwideisonet import *
from models.superwideresnet import *
from datasets import *
from torch.utils import data
from mixup import *
import argparse
import sys
import random
import numpy
import time
from comm_utils import insert_mimo_channel


# ==================================================================================================
# FUNCTIONS
# ==================================================================================================
def compute_accuracy():
    correct = 0
    total = 0
    time_tot = 0

    with torch.no_grad():
        for inputs, labels in evalloader:
            if inputs.shape[0] != args.num * args.batch_size:
                continue
            # transfer data to GPUs
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # calculate outputs by running images through the network
            time_start = time.time()
            outputs = model(inputs)
            time_end = time.time()

            effective_batch_size = outputs.shape[0] # due to superposition batch may be truncated
            labels = labels[:effective_batch_size]
            
            # statistics
            _, predicted = torch.max(outputs, 1)
            total += effective_batch_size
            correct += (predicted == labels).sum().item()

            time_tot += (time_end-time_start)
            
        accuracy = 100 * correct / total
        time_tot = time_tot / total

        return accuracy, time_tot

if __name__ == '__main__': # avoids rerunning code when multiple processes are spawned (for quicker dataloading)

    #------------- reproducibility ---------------  
    random.seed(0)
    numpy.random.seed(0)
    torch.manual_seed(0)
    # seed dataloader workers
    def seed_worker(worker_id):
        numpy.random.seed(0)
        random.seed(0)
    # seed generators
    g = torch.Generator()
    g.manual_seed(0)

    #------------- argument parsing --------------- 
    parser = argparse.ArgumentParser(description='Evaluates CNNs, in particular demonstrating superposition principles', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('model', type=str, choices=["WideResNet-28", "WideISOReLUNet-28", "WideISONet-28", "MIMONet-28", "WideResNet-16", "WideISOReLUNet-16", "WideISONet-16", "MIMONet-16", "MIMONet-10", "MIMODistNet-10", "MIMODistNet-16", "MIMODistNet-28"], help='architecture and network depth')
    parser.add_argument('dataset', type=str, choices=["CIFAR10", "CIFAR100", "MNIST", "CUB"], help='dataset')
    parser.add_argument("type", type=str, choices=["None", "HRR", "MBAT"], help="binding type")
    parser.add_argument("num", type=int, help="maximum superposition capability of model")

    parser.add_argument("-w", "--width", type=int, default=1, help="width of network, i.e. factor to increase channel size")
    parser.add_argument("-i", "--initial_width", type=int, default=1, help="width of output of initial convolutional layer, which is not affected by [width]")
    parser.add_argument('-p', "--relu_parameter_init", type=float, default=None, help="offset in shiftedReLU (default -1) or parameter in parametricReLU (default 0.5) depending on the model")
    parser.add_argument("-k", "--skip_init", action="store_true", help="whether skip_init (ignore residual branch at initialisation) is enabled")
    parser.add_argument("-q", "--dirac_init", action="store_true", help="whether dirac initialisation of convolutions is enabled")
    parser.add_argument("-z", "--batch_norm_disabled", action="store_true", help="whether batch norm is disabled")
    parser.add_argument("-v", "--trainable_keys_disabled", action="store_true", help="whether keys are fixed after initialisation")

    parser.add_argument('-s', "--sup_low", type=int, default=4, help="how many images are superposed in the low-demand setting. If left unspecified always [num] images are superposed")

    parser.add_argument("-b", "--batch_size", type=int, default=1024, help="the batch size used. The batch size before binding is larger by a factor [num]")
    parser.add_argument("-n", "--number_of_cpus", type=int, default=16, help="number of cpus used in dataloading")
    parser.add_argument("-c", "--checkpoint", type=str, default="", help="which model weights to load")

    parser.add_argument('--split_layer', type=str, default=None, help="the splitting layer between the head and tail of the distributed DNN")
    parser.add_argument('--comm_n_streams', type=int, default=8, help="the number of data streams in MIMO communication system")
    parser.add_argument('--channel_model', type=str, default="awgn", help="the model for the MIMO communication channel")
    parser.add_argument('--comm_snr', type=float, default=20, help="the SNR for the additive white Gaussian noise communication channel used for training")
    parser.add_argument('--precoder_type', type=str, default="task-oriented", help="the MIMO communication precoding technique")
 
    args = parser.parse_args()
    #------------- process settings ---------------
    sup_low = None
    if args.sup_low:
        if args.sup_low > args.num or args.sup_low < 1 or args.num % args.sup_low != 0: 
            print(f'option --sup_low (-s) must divide argument num')
            sys.exit(2)
        else:
            sup_low = args.sup_low
    else:
        sup_low = args.num

    if args.dataset == "CIFAR10" :
        num_classes = 10
        input_channels = 3
    elif args.dataset == "MNIST":
        num_classes = 10
        input_channels = 1
    elif args.dataset == "CIFAR100":
        num_classes = 100
        input_channels = 3
    elif args.dataset == "CUB":
        num_classes = 200
        input_channels = 3
    else:
        print(f'unknown argument {args.dataset} for dataset')
        sys.exit(2)

    if args.batch_norm_disabled:
        norm = IdentityNorm
    else:
        norm = None # defaults to BatchNorm

    model = {"WideResNet-16":SuperWideResnet(num_img_sup_cap = args.num, binding_type = args.type, width=args.width, layers= [2, 2, 2], initial_width=args.initial_width, num_classes=num_classes, norm_layer=norm, skip_init=args.skip_init, trainable_keys = not args.trainable_keys_disabled),
             "WideISONet-16":SuperWideISONet(num_img_sup_cap = args.num, binding_type = args.type, width=args.width, layers= [2, 2, 2], initial_width=args.initial_width, num_classes=num_classes, norm_layer=norm, block=BasicISOBlock, dirac_init=args.dirac_init, relu_parameter=args.relu_parameter_init, skip_init=args.skip_init, trainable_keys = not args.trainable_keys_disabled, input_channels = input_channels),
             "WideISOReLUNet-16":SuperWideISONet(num_img_sup_cap = args.num, binding_type = args.type, width=args.width, layers= [2, 2, 2], initial_width=args.initial_width, num_classes=num_classes, norm_layer=norm, block=BasicBlock, dirac_init=args.dirac_init, relu_parameter=args.relu_parameter_init, skip_init=args.skip_init, trainable_keys = not args.trainable_keys_disabled, input_channels = input_channels),
             "MIMONet-10":SuperWideISONet(num_img_sup_cap = args.num, binding_type = args.type, width=args.width, layers= [1, 1, 1], initial_width=args.initial_width, num_classes=num_classes, norm_layer=norm, block=AdjustedISOBlock, dirac_init=args.dirac_init, relu_parameter=args.relu_parameter_init, skip_init=args.skip_init, trainable_keys = not args.trainable_keys_disabled, input_channels = input_channels),
             "MIMONet-16":SuperWideISONet(num_img_sup_cap = args.num, binding_type = args.type, width=args.width, layers= [2, 2, 2], initial_width=args.initial_width, num_classes=num_classes, norm_layer=norm, block=AdjustedISOBlock, dirac_init=args.dirac_init, relu_parameter=args.relu_parameter_init, skip_init=args.skip_init, trainable_keys = not args.trainable_keys_disabled, input_channels = input_channels),
             "WideResNet-28":SuperWideResnet(num_img_sup_cap = args.num, binding_type = args.type, width=args.width, layers= [4, 4, 4], initial_width=args.initial_width, num_classes=num_classes, norm_layer=norm, skip_init=args.skip_init, trainable_keys = not args.trainable_keys_disabled, input_channels = input_channels),
             "WideISONet-28":SuperWideISONet(num_img_sup_cap = args.num, binding_type = args.type, width=args.width, layers= [4, 4, 4], initial_width=args.initial_width, num_classes=num_classes, norm_layer=norm, block=BasicISOBlock, dirac_init=args.dirac_init, relu_parameter=args.relu_parameter_init, skip_init=args.skip_init, trainable_keys = not args.trainable_keys_disabled, input_channels = input_channels),
             "WideISOReLUNet-28":SuperWideISONet(num_img_sup_cap = args.num, binding_type = args.type, width=args.width, layers= [4, 4, 4], initial_width=args.initial_width, num_classes=num_classes, norm_layer=norm, block=BasicBlock, dirac_init=args.dirac_init, relu_parameter=args.relu_parameter_init, skip_init=args.skip_init, trainable_keys = not args.trainable_keys_disabled, input_channels = input_channels),
             "MIMONet-28":SuperWideISONet(num_img_sup_cap = args.num, binding_type = args.type, width=args.width, layers= [4, 4, 4], initial_width=args.initial_width, num_classes=num_classes, norm_layer=norm, block=AdjustedISOBlock, dirac_init=args.dirac_init, relu_parameter=args.relu_parameter_init, skip_init=args.skip_init, trainable_keys = not args.trainable_keys_disabled, input_channels = input_channels),
             "MIMODistNet-10":SuperWideISONet(num_img_sup_cap = args.num, binding_type = args.type, width=args.width, layers= [1, 1, 1], initial_width=args.initial_width, num_classes=num_classes, norm_layer=norm, block=AdjustedISOBlock, dirac_init=args.dirac_init, relu_parameter=args.relu_parameter_init, skip_init=args.skip_init, trainable_keys = not args.trainable_keys_disabled, input_channels = input_channels),
             "MIMODistNet-16":SuperWideISONet(num_img_sup_cap = args.num, binding_type = args.type, width=args.width, layers= [2, 2, 2], initial_width=args.initial_width, num_classes=num_classes, norm_layer=norm, block=AdjustedISOBlock, dirac_init=args.dirac_init, relu_parameter=args.relu_parameter_init, skip_init=args.skip_init, trainable_keys = not args.trainable_keys_disabled, input_channels = input_channels),
             "MIMODistNet-28":SuperWideISONet(num_img_sup_cap = args.num, binding_type = args.type, width=args.width, layers= [4, 4, 4], initial_width=args.initial_width, num_classes=num_classes, norm_layer=norm, block=AdjustedISOBlock, dirac_init=args.dirac_init, relu_parameter=args.relu_parameter_init, skip_init=args.skip_init, trainable_keys = not args.trainable_keys_disabled, input_channels = input_channels)
             }.get(args.model)
    if model == None:
        print(f'unknown argument {args.model} for model')
        sys.exit(2)

    if "Dist" in args.model and args.precoder_type == "task-oriented":
        insert_mimo_channel(model, split_layer=args.split_layer, n_streams=args.comm_n_streams, channel_model=args.channel_model, snr=args.comm_snr, precoder_type=args.precoder_type, batch_size=args.batch_size)

    cuda_available = torch.cuda.is_available()
    device = 'cuda' if cuda_available else 'cpu'
    pin_memory = True if cuda_available else False # pin memory may speed up inference by allowing faster data loading from dedicated main memory, depends on system used.

    checkpoint = torch.load(args.checkpoint, map_location=torch.device(device))
    model.load_state_dict(checkpoint['model_state_dict'])

    #------------- finished parsing arguments and fixing settings ---------------
    _, evalset = get_train_eval_test_sets(args.dataset, True)

    evalloader = data.DataLoader(evalset, batch_size=args.batch_size*args.num,
                                    shuffle=True, num_workers=args.number_of_cpus, pin_memory=pin_memory, drop_last=False, worker_init_fn=seed_worker, generator=g)
    
    if "Dist" in args.model and args.precoder_type in ["ZF", "SVD"]:
        insert_mimo_channel(model, split_layer=args.split_layer, n_streams=args.comm_n_streams, channel_model=args.channel_model, snr=args.comm_snr, precoder_type=args.precoder_type, batch_size=args.batch_size)

    model = model.to(device)

    model.num_img_sup = sup_low

    if "Dist" in args.model:
        for i in range(10):
            accuracy, time_tot = compute_accuracy()
            print(f'Accuracy with {sup_low} images superposed at superposition capacity {model.num_img_sup_cap}: {accuracy}\nTime per sample: {time_tot}')
            print(f'Updating the channel matrix')
            model.get_submodule(f"{args.split_layer}.2").update_channel_matrix()
            model.get_submodule(f"{args.split_layer}.2").to(device)
            model.get_submodule(f"{args.split_layer}.1").set_channel_matrix(model.get_submodule(f"{args.split_layer}.2").channel_matrix)
            model.get_submodule(f"{args.split_layer}.1").to(device)
            model.get_submodule(f"{args.split_layer}.3").set_channel_matrix(model.get_submodule(f"{args.split_layer}.2").channel_matrix)
            model.get_submodule(f"{args.split_layer}.3").to(device)
    else:
        accuracy, time_tot = compute_accuracy()

        print(f'Accuracy with {sup_low} images superposed at superposition capacity {model.num_img_sup_cap}: {accuracy}\nTime per sample: {time_tot}')