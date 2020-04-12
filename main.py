import os
import math
import argparse
import tensorboardX
import torch.optim as optim
import horovod.torch as hvd
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed

from tqdm import tqdm

from dataloaders import *
from torchvision.models import alexnet
from optimizer import QSGDCompressor
from optimizer import NearestNeighborCompressor
from optimizer.distributed_compressed_optimizer import DistributedNNQOptimizer
data_loaders = {
    'mnist':    minst,
    'cifar10':  cifar10,
    'cifar100': cifar100,
    'imagenet': imagenet,
}

classes_choices = {
    'mnist':    10,
    'cifar10':  10,
    'cifar100': 100,
    'imagenet': 1000
}


def setup_network(args):
    if args.dataset == 'imagenet':
        from torchvision.models import resnet18, resnet34, resnet50,\
            resnet101, resnet152, vgg11, vgg13, vgg16, vgg19
    else:
        from models import resnet18, resnet34, resnet50, \
            resnet101, resnet152, vgg11, vgg13, vgg16, vgg19

    network_choices = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152,
        'vgg11': vgg11,
        'vgg13': vgg13,
        'vgg16': vgg16,
        'vgg19': vgg19,
        'alexnet': alexnet,
    }

    NETWORK = network_choices[args.network]
    args.num_classes = classes_choices[args.dataset]
    args.loader = data_loaders[args.dataset]
    return NETWORK()


# Training settings
parser = argparse.ArgumentParser(
    description='PyTorch ImageNet Example',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--checkpoint-format',
                    default='./checkpoint-{epoch}.pth.tar',
                    help='checkpoint file format')
parser.add_argument('--fp16-allreduce',
                    action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--batches-per-allreduce', type=int, default=1,
                    help='number of batches processed locally before'
                         ' executing allreduce across workers; it '
                         'multiplies total batch size.')

parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=32,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=90,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.0125,
                    help='learning rate for a single GPU')
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.00005,
                    help='weight decay')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')

parser.add_argument('--network', type=str, default="resnet50")
parser.add_argument('--dataset', type=str, default="imagenet")

parser.add_argument('--c-dim', type=int, default=8)
parser.add_argument('--k-bit', type=int, default=8)
parser.add_argument('--n-bit', type=int, default=8)


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


hvd.init()
torch.manual_seed(args.seed)

if args.cuda:
    # Horovod: pin GPU to local rank.
    torch.cuda.set_device(hvd.local_rank())
    torch.cuda.manual_seed(args.seed)

cudnn.benchmark = True

# If set > 0, will resume training from a given checkpoint.
resume_from_epoch = 0
for i_epoch in range(args.epochs, 0, -1):
    if os.path.exists(args.checkpoint_format.format(epoch=i_epoch)):
        resume_from_epoch = i_epoch
        break

resume_from_epoch = hvd.broadcast(torch.tensor(resume_from_epoch),
                                  root_rank=0,
                                  name='resume_from_epoch').item()

verbose = 1 if hvd.rank() == 0 else 0
log_writer = None if hvd.rank() != 0 else \
    tensorboardX.SummaryWriter(args.log_dir)


model = setup_network(args)

if args.cuda:
    model.cuda()

allreduce_batch_size = args.batch_size * args.batches_per_allreduce

kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
train_dataset, val_dataset = args.loader()
train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=allreduce_batch_size,
    sampler=train_sampler, **kwargs)

val_sampler = torch.utils.data.distributed.DistributedSampler(
    val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.val_batch_size,
    sampler=val_sampler, **kwargs)



optimizer = optim.SGD(model.parameters(),
                      lr=(args.base_lr *
                          args.batches_per_allreduce * hvd.size()),
                      momentum=args.momentum, weight_decay=args.wd)
# compression = hvd.Compression.fp16
compression = hvd.Compression.none
# args.compressor = QSGDCompressor
# optimizer = DistributedNNQOptimizer(
#     args, optimizer,
#     named_parameters=model.named_parameters(),
#     compression=compression,
#     backward_passes_per_step=args.batches_per_allreduce)
optimizer = hvd.DistributedOptimizer(
   optimizer,
   named_parameters=model.named_parameters(),
   compression=compression,
   backward_passes_per_step=args.batches_per_allreduce)


if resume_from_epoch > 0 and hvd.rank() == 0:
    filepath = args.checkpoint_format.format(epoch=resume_from_epoch)
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)


def train(epoch):
    model.train()
    train_sampler.set_epoch(epoch)
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')

    with tqdm(total=len(train_loader),
              desc='Train Epoch     #{}'.format(epoch + 1),
              disable=not verbose) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            adjust_learning_rate(epoch, batch_idx)

            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            # Split data into sub-batches of size batch_size
            for i in range(0, len(data), args.batch_size):
                data_batch = data[i:i + args.batch_size]
                target_batch = target[i:i + args.batch_size]
                output = model(data_batch)
                train_accuracy.update(accuracy(output, target_batch))
                loss = F.cross_entropy(output, target_batch)
                train_loss.update(loss)
                # Average gradients among sub-batches
                loss.div_(math.ceil(float(len(data)) / args.batch_size))
                loss.backward()
            # Gradient is applied across all ranks
            optimizer.step()
            t.set_postfix({'average loss': train_loss.avg.item(),
                           'batched loss': train_loss.val.item(),
                           'average acc ': 100. *
                                           train_accuracy.avg.item(),
                           'batched acc ': 100. *
                                           train_accuracy.val.item()
                           })
            t.update(1)

    if log_writer:
        log_writer.add_scalar('train/loss', train_loss.avg, epoch)
        log_writer.add_scalar('train/accuracy', train_accuracy.avg, epoch)


def validate(epoch):
    model.eval()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')

    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)

                val_loss.update(F.cross_entropy(output, target))
                val_accuracy.update(accuracy(output, target))
                t.set_postfix({'average loss': val_loss.avg.item(),
                               'batched loss': val_loss.val.item(),
                               'average acc ': 100. *
                                               val_accuracy.avg.item(),
                               'batched acc ': 100. *
                                               val_accuracy.val.item()
                               })
                t.update(1)

    if log_writer:
        log_writer.add_scalar('val/loss', val_loss.avg, epoch)
        log_writer.add_scalar('val/accuracy',
                              val_accuracy.avg, epoch)


def adjust_learning_rate(epoch, batch_idx):
    if epoch < args.warmup_epochs:
        epoch += float(batch_idx + 1) / len(train_loader)
        lr_adj = 1. / hvd.size() * (epoch * (hvd.size() - 1)
                                    / args.warmup_epochs + 1)
    elif epoch < 30:
        lr_adj = 1.
    elif epoch < 60:
        lr_adj = 1e-1
    elif epoch < 80:
        lr_adj = 1e-2
    else:
        lr_adj = 1e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * hvd.size() * \
                            args.batches_per_allreduce * lr_adj


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


def save_checkpoint(epoch):
    if hvd.rank() == 0:
        filepath = args.checkpoint_format.format(epoch=epoch + 1)
        state = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(state, filepath)


class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)
        self.val = None

    def update(self, val):
        self.sum += hvd.allreduce(val.detach().cpu(), name=self.name)
        self.n += 1
        self.val = val

    @property
    def avg(self):
        return self.sum / self.n

    @property
    def value(self):
        return self.val

for epoch in range(resume_from_epoch, args.epochs):
    train(epoch)
    validate(epoch)
    save_checkpoint(epoch)

