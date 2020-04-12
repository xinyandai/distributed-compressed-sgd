import torch
from torchvision import datasets, transforms
from preprocess import inception_preproccess, \
    scale_crop, pad_random_crop


__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                   'std': [0.229, 0.224, 0.225]}


def minst():
    normalize = {'mean': [0.5], 'std': [0.5]}
    transform_train = pad_random_crop(
        input_size=28, scale_size=32, normalize=normalize)
    transform_test = scale_crop(
        input_size=28, scale_size=28, normalize=normalize)

    train_dataset = datasets.MNIST(
        './data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.MNIST(
        './data', train=False, transform=transform_test)
    return train_dataset, test_dataset


def cifar10():
    transform_train = pad_random_crop(
        input_size=32, scale_size=40, normalize=__imagenet_stats)

    transform_test = scale_crop(
        input_size=32, scale_size=32, normalize=__imagenet_stats)

    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)

    return train_dataset, test_dataset


def cifar100():
    transform_train = pad_random_crop(
        input_size=32, scale_size=40, normalize=__imagenet_stats)

    transform_test = scale_crop(
        input_size=32, scale_size=32, normalize=__imagenet_stats)

    train_dataset = datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    return train_dataset, test_dataset



def imagenet():
    scale_size =  256
    input_size =  224

    transform_train = inception_preproccess(input_size = input_size,
                                            normalize=__imagenet_stats)
    transform_test =  scale_crop(input_size=input_size,
                                 scale_size=scale_size,
                                 normalize=__imagenet_stats)

    traindir = '/data/dataset/ILSVRC2012/ILSVRC2012_img_train'
    testdir = '/data/dataset/ILSVRC2012/ILSVRC2012_img_val'


    train_dataset = datasets.ImageFolder(traindir, transform_train)
    test_dataset = datasets.ImageFolder(testdir, transform_test)

    return train_dataset, test_dataset

