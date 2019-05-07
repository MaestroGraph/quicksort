# import torch
# from torch.utils.data.dataset import Dataset
#
# import torchvision
# from torchvision.transforms import ToTensor
# from torch.utils.data import TensorDataset, DataLoader
# import util
#
#
# from enum import Enum
#
# BATCH = 32
# TRAIN_SIZE = 55_000
# VAL_SIZE = 5_000
# TEST_SIZE = 10_000
#
# class Mode(Enum):
#     TRAIN = 1  # training set only
#     VAL = 2  # validation only
#     TRAINVAL = 3  # training + validation (for the final training run)
#     TEST = 4  #
#
# class MNISTCatDataset(Dataset):
#     """
#     Dataset containing a fixed number of randomly generated n-digit mnist numbers, created by concatenating random
#     instances from the MNIST data.
#     """
#
#     def __init__(self, width=4, mode=Mode.TRAIN, data='./data', seed=0):
#
#         self.mode = mode
#         self.width = width
#         self.seed = seed
#
#         if mode == Mode.TRAIN:
#             images = torchvision.datasets.MNIST(root=data, train=True, download=True, transform=ToTensor())
#             self.loader = DataLoader(images, batch_size=BATCH, sampler=util.ChunkSampler(0, TRAIN_SIZE, TRAIN_SIZE+VAL_SIZE))
#         elif mode == Mode.VAL:
#             images = torchvision.datasets.MNIST(root=data, train=True, download=True, transform=ToTensor())
#             self.loader = DataLoader(images, batch_size=BATCH, sampler=util.ChunkSampler(TRAIN_SIZE, VAL_SIZE, TRAIN_SIZE+VAL_SIZE))
#         elif mode == Mode.TRAINVAL
#             images = torchvision.datasets.MNIST(root=data, train=True, download=True, transform=ToTensor())
#             self.loader = torch.utils.data.DataLoader(images, batch_size=BATCH, shuffle=True, num_workers=2)
#         elif mode == Mode.TEST:
#             images = torchvision.datasets.MNIST(root=data, train=False, download=True, transform=ToTensor())
#             self.loader = torch.utils.data.DataLoader(images, batch_size=BATCH, shuffle=False, num_workers=2)
#         else:
#             assert False
#
#
#
#     def __getitem__(self, index):
#         # stuff
#         data =  # Some data read from a file or image
#         if self.transforms is not None:
#             data = self.transforms(data)
#         # If the transform variable is not empty
#         # then it applies the operations in the transforms with the order that it is created.
#         return (img, label)
#
#     def __len__(self):
#
#         if   self.mode == Mode.TRAIN:
#             return TRAIN_SIZE
#         elif self.mode = Mode.VAL:
#             return VAL_SIZE
#         elif self.mode = Mode.TRAINVAL:
#             return TRAIN_SIZE + VAL_SIZE
#         elif self.mode = Mode.TEST:
#             return TEST_SIZE
#
#         assert False