#################### Imports #########################
from __future__ import print_function, division
import argparse
import os
import json
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.models as models
import torch.nn as nn
from torch.optim import lr_scheduler
import time
import numpy as np
from torch.autograd import Variable
from pathlib import Path
from PIL import Image
import pandas as pd
#from skimage import io, transform
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import time
from IPython.display import clear_output

import torch.nn as nn
import torch.nn.functional as F


##################################################

# This script was the first attempt at gathering 
# the necassary code to appropriately run the 
# training of SegNet. The script was made to
# save the model as well as be flexible for 
# running in parallel over multiple instances.

# See refactored segnet2 training script

##################################################


def _average_gradients(model):
    # Gradient averaging.
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM, group=0)
        param.grad.data /= size


class SegNet(nn.Module):
    """SegNet: A Deep Convolutional Encoder-Decoder Architecture for
    Image Segmentation. https://arxiv.org/abs/1511.00561
    See https://github.com/alexgkendall/SegNet-Tutorial for original models.
    Args:
        num_classes (int): number of classes to segment
        n_init_features (int): number of input features in the fist convolution
        drop_rate (float): dropout rate of each encoder/decoder module
        filter_config (list of 5 ints): number of output features at each level
    """
    def __init__(self, num_classes, n_init_features=1, drop_rate=0.5,
                 filter_config=(64, 128, 256, 512, 512)):
        super(SegNet, self).__init__()

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        # setup number of conv-bn-relu blocks per module and number of filters
        encoder_n_layers = (2, 2, 3, 3, 3)
        encoder_filter_config = (n_init_features,) + filter_config
        decoder_n_layers = (3, 3, 3, 2, 1)
        decoder_filter_config = filter_config[::-1] + (filter_config[0],)

        for i in range(0, 5):
            # encoder architecture
            self.encoders.append(_Encoder(encoder_filter_config[i],
                                          encoder_filter_config[i + 1],
                                          encoder_n_layers[i], drop_rate))

            # decoder architecture
            self.decoders.append(_Decoder(decoder_filter_config[i],
                                          decoder_filter_config[i + 1],
                                          decoder_n_layers[i], drop_rate))

        # final classifier (equivalent to a fully connected layer)
        self.classifier = nn.Conv2d(filter_config[0], num_classes, 3, 1, 1)

    def forward(self, x):
        indices = []
        unpool_sizes = []
        feat = x

        # encoder path, keep track of pooling indices and features size
        for i in range(0, 5):
            (feat, ind), size = self.encoders[i](feat)
            indices.append(ind)
            unpool_sizes.append(size)

        # decoder path, upsampling with corresponding indices and size
        for i in range(0, 5):
            feat = self.decoders[i](feat, indices[4 - i], unpool_sizes[4 - i])

        return self.classifier(feat)


class _Encoder(nn.Module):
    def __init__(self, n_in_feat, n_out_feat, n_blocks=2, drop_rate=0.5):
        """Encoder layer follows VGG rules + keeps pooling indices
        Args:
            n_in_feat (int): number of input features
            n_out_feat (int): number of output features
            n_blocks (int): number of conv-batch-relu block inside the encoder
            drop_rate (float): dropout rate to use
        """
        super(_Encoder, self).__init__()

        layers = [nn.Conv2d(n_in_feat, n_out_feat, 3, 1, 1),
                  nn.BatchNorm2d(n_out_feat),
                  nn.ReLU(inplace=True)]

        if n_blocks > 1:
            layers += [nn.Conv2d(n_out_feat, n_out_feat, 3, 1, 1),
                       nn.BatchNorm2d(n_out_feat),
                       nn.ReLU(inplace=True)]
            if n_blocks == 3:
                layers += [nn.Dropout(drop_rate)]

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        output = self.features(x)
        return F.max_pool2d(output, 2, 2, return_indices=True), output.size()


class _Decoder(nn.Module):
    """Decoder layer decodes the features by unpooling with respect to
    the pooling indices of the corresponding decoder part.
    Args:
        n_in_feat (int): number of input features
        n_out_feat (int): number of output features
        n_blocks (int): number of conv-batch-relu block inside the decoder
        drop_rate (float): dropout rate to use
    """
    def __init__(self, n_in_feat, n_out_feat, n_blocks=2, drop_rate=0.5):
        super(_Decoder, self).__init__()

        layers = [nn.Conv2d(n_in_feat, n_in_feat, 3, 1, 1),
                  nn.BatchNorm2d(n_in_feat),
                  nn.ReLU(inplace=True)]

        if n_blocks > 1:
            layers += [nn.Conv2d(n_in_feat, n_out_feat, 3, 1, 1),
                       nn.BatchNorm2d(n_out_feat),
                       nn.ReLU(inplace=True)]
            if n_blocks == 3:
                layers += [nn.Dropout(drop_rate)]

        self.features = nn.Sequential(*layers)

    def forward(self, x, indices, size):
        unpooled = F.max_unpool2d(x, indices, 2, 2, 0, size)
        return self.features(unpooled)



if __name__ =='__main__':

    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--use-cuda', type=bool, default=False)

    # Data, model, and output directories
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    # parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])                               don't need test right now, can test in seperate script

    # default to the value in environment variable `SM_MODEL_DIR`. Using args makes the script more portable.
    #parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])


    parser.add_argument('--backend', type=str, default=None,
                        help='backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)')

    # Container environment
    #env = sagemaker_containers.training_env()
    #parser.add_argument('--hosts', type=list, default=env.hosts)

    # using it in argparse
    parser.add_argument('hosts', type=str, default=json.loads(os.environ['SM_HOSTS']))

    #parser.add_argument('--current-host', type=str, default=env.current_host)
    # using it in argparse
    parser.add_argument('--current_host', type=str, default=os.environ['SM_CURRENT_HOST'])
    #parser.add_argument('--num-gpus', type=int, default=env.num_gpus)
    # using it in argparse
    parser.add_argument('--num_gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    # using it as variable
    num_gpus = int(os.environ['SM_NUM_GPUS'])
    # using it as variable
    hosts = json.loads(os.environ['SM_HOSTS'])
    
    args, _ = parser.parse_known_args()
    
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    #logger.debug("Distributed training - {}".format(is_distributed))
    use_cuda = num_gpus > 0
    #logger.debug("Number of gpus available - {}".format(args.num_gpus))
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    device = torch.device("cuda" if use_cuda else "cpu")

    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(hosts)
        os.environ['WORLD_SIZE'] = str(world_size)
        # using it as variable
        current_host = os.environ['SM_CURRENT_HOST']
        ######## breaking the code not sure what it was in the first place
        #host_rank = args.hosts.index(current_host)
        #dist.init_process_group(backend=args.backend, 
                                #rank=host_rank, 
                               # world_size=world_size)
        #logger.info(
           # 'Init distributed env: \'{}\' backend on {} nodes. '.format(args.backend, 
            #    dist.get_world_size()) + \
           # 'Current host rank is {}. Number of gpus: {}'.format(
            #    dist.get_rank(), args.num_gpus))
        

    ################################# Training ###################################
    # ... load from args.train and args.test, train a model, write model to args.model_dir.

    ######### Unpack Args ##########
    train_dir = args.train
    #model_dir = args.model_dir                                                                          just copy and pasted model becuase it wasn't working

    batch_size = args.batch_size
    learning_rate = args.learning_rate
    epochs = args.epochs

    use_cuda = args.use_cuda



    ############## suspect that this is causing the weird tensor bug
    ########## GPU stuff that I don't know lol #############
    # define variables if GPU is to be used
    #if torch.cuda.is_available():
    #    use_gpu = True
    #    print("Using GPU")
    #else:
    #    use_gpu = False
    #FloatTensor = torch.cuda.FloatTensor if use_gpu else torch.FloatTensor
    #LongTensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor
    #ByteTensor = torch.cuda.ByteTensor if use_gpu else torch.ByteTensor
    #Tensor = FloatTensor

    ########## Dataclass for segmentation ###########
    class CloudDataset(Dataset):
        def __init__(self, r_dir, g_dir, b_dir, nir_dir, gt_dir, pytorch=True):
            super().__init__()

            # Loop through the files in red folder and combine, into a dictionary, the other bands
            self.files = [self.combine_files(f, g_dir, b_dir, nir_dir, gt_dir) for f in r_dir.iterdir() if
                          not f.is_dir()]
            self.pytorch = pytorch

        def combine_files(self, r_file: Path, g_dir, b_dir, nir_dir, gt_dir):

            files = {'red': r_file,
                     'green': g_dir / r_file.name.replace('red', 'green'),
                     'blue': b_dir / r_file.name.replace('red', 'blue'),
                     'nir': nir_dir / r_file.name.replace('red', 'nir'),
                     'gt': gt_dir / r_file.name.replace('red', 'gt')}

            return files

        def __len__(self):

            return len(self.files)

        def open_as_array(self, idx, invert=False, include_nir=False):

            raw_rgb = np.stack([np.array(Image.open(self.files[idx]['red'])),
                                np.array(Image.open(self.files[idx]['green'])),
                                np.array(Image.open(self.files[idx]['blue'])),
                                ], axis=2)

            if include_nir:
                nir = np.expand_dims(np.array(Image.open(self.files[idx]['nir'])), 2)
                raw_rgb = np.concatenate([raw_rgb, nir], axis=2)

            if invert:
                raw_rgb = raw_rgb.transpose((2, 0, 1))

            # normalize
            return (raw_rgb / np.iinfo(raw_rgb.dtype).max)

        def open_mask(self, idx, add_dims=False):

            raw_mask = np.array(Image.open(self.files[idx]['gt']))
            raw_mask = np.where(raw_mask == 255, 1, 0)

            return np.expand_dims(raw_mask, 0) if add_dims else raw_mask

        def __getitem__(self, idx):

            x = torch.tensor(self.open_as_array(idx, invert=self.pytorch, include_nir=True), dtype=torch.float32)
            y = torch.tensor(self.open_mask(idx, add_dims=False), dtype=torch.torch.int64)

            return x, y

        def open_as_pil(self, idx):

            arr = 256 * self.open_as_array(idx)

            return Image.fromarray(arr.astype(np.uint8), 'RGB')

        def __repr__(self):
            s = 'Dataset class with {} files'.format(self.__len__())

            return s

    ########## Dataclass for segmentation ###########
    base_path = Path(train_dir)                                     #### NEW CHANGE FOR SAGE
    data = CloudDataset(base_path / 'train_red',
                        base_path / 'train_green',
                        base_path / 'train_blue',
                        base_path / 'train_nir',
                        base_path / 'train_gt')

    ####### Split the data ########
    train_ds, valid_ds = torch.utils.data.random_split(data, (6000, 2400))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)


    ################ Initialise Model ###############
    num_classes = 2  # assuming cloud and non cloud
    num_channels = 4  # for the cloud data, for now
    model = SegNet(num_classes, n_init_features=num_channels)
    model = model.to(device)
    
        
    
    if is_distributed and use_cuda:
        # multi-machine multi-gpu case
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # single-machine multi-gpu case or single-machine or multi-machine cpu case
        model = torch.nn.DataParallel(model)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # using this becuase SmokeNet did
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    ################ Model training ##############

    def train(model, train_dl, valid_dl, loss_fn, optimizer, scheduler, acc_fn, epochs=1):
        start = time.time()

        train_loss, valid_loss = [], []

        best_acc = 0.0

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)

            for phase in ['train', 'valid']:
                if phase == 'train':
                    model.train(True)  # Set trainind mode = true
                    dataloader = train_dl
                else:
                    model.train(False)  # Set model to evaluate mode
                    dataloader = valid_dl

                running_loss = 0.0
                running_acc = 0.0

                step = 0

                # iterate over data
                for x, y in dataloader:
                    x = x.to(device)
                    y = y.to(device)
                    
                    step += 1

                    # forward pass
                    if phase == 'train':
                        # zero the gradients
                        optimizer.zero_grad()
                        outputs = model(x)
                        loss = loss_fn(outputs, y)

                        # the backward pass frees the graph memory, so there is no
                        # need for torch.no_grad in this training pass
                        loss.backward()
                        if is_distributed and not use_cuda:
                        # average gradients manually for multi-machine cpu case only
                            _average_gradients(model)
                        optimizer.step()
                        scheduler.step()

                    else:
                        with torch.no_grad():
                            outputs = model(x)
                            loss = loss_fn(outputs, y.long())

                    # stats - whatever is the phase
                    acc = acc_fn(outputs, y)

                    running_acc += acc * dataloader.batch_size
                    running_loss += loss * dataloader.batch_size

                    if step % 100 == 0:
                        # clear_output(wait=True)
                        print('Current step: {}  Loss: {}  Acc: {}  AllocMem (Mb): {}'.format(step, loss, acc,
                                                                                              torch.cuda.memory_allocated() / 1024 / 1024))
                        # print(torch.cuda.memory_summary())

                epoch_loss = running_loss / len(dataloader.dataset)
                epoch_acc = running_acc / len(dataloader.dataset)

                clear_output(wait=True)
                print('Epoch {}/{}'.format(epoch, epochs - 1))
                print('-' * 10)
                print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))
                print('-' * 10)

                train_loss.append(epoch_loss) if phase == 'train' else valid_loss.append(epoch_loss)

        time_elapsed = time.time() - start
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        return train_loss, valid_loss


    def acc_metric(predb, yb):
        return (predb.argmax(dim=1) == yb.cuda()).float().mean()
    
    '''
    if torch.cuda.is_available():
      model.cuda()

    if torch.cuda.device_count() > 1:
      print("Using", torch.cuda.device_count(), "GPUs!")
      # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
      model = nn.DataParallel(model)
    '''
    
    
    ####### train
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=learning_rate)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    train_loss, valid_loss = train(model, train_dl, valid_dl, loss_fn, opt,exp_lr_scheduler, acc_metric, epochs=epochs)


    ############################## Save model ####################################
    # ... train `model`, then save it to `model_dir`
    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        #logger.info("Saving the model.")
        torch.save(model.cpu().state_dict(), f)








