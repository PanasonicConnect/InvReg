import os
from torch.autograd import Function
import torch.distributed as dist
import torch


class DistAllGather(Function):
    @staticmethod
    def forward(ctx, input):
        wsize = int(os.environ["WORLD_SIZE"])
        ctx.save_for_backward(input)
        input = input.clone()
        gather_list = [torch.zeros_like(input) for i in range(wsize)]
        dist.barrier()
        dist.all_gather(gather_list, input, dist.group.WORLD)
        return tuple(gather_list)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        dist.barrier()
        dist.reduce_scatter(grad_out, list(grads), group=dist.group.WORLD)
        return grad_out


class AllGather(torch.nn.Module):
    def __init__(self):
        super(AllGather, self).__init__()

    def forward(self, input):
        return DistAllGather.apply(input)


class DistAllGatherSlice(Function):

    @staticmethod
    def forward(ctx, input):
        wsize = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        wrank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        ctx.save_for_backward(input)
        input = input.clone()
        rows = input.shape[0] // wsize
        gather_list_tmp = [torch.zeros_like(input[:rows]) for i in range(wsize)]
        for i in range(wsize):
            dist.barrier()
            dist.all_gather(gather_list_tmp, input[rows * i:rows * (i + 1)], group=dist.group.WORLD)
            if i == wrank:
                gather_list = [g.clone() for g in gather_list_tmp]
        return tuple(gather_list)

    @staticmethod
    def backward(ctx, *grads):
        wsize = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        wrank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        rows = grad_out.shape[0] // wsize
        gather_list = [torch.zeros_like(grads[0]) for i in range(wsize)]
        for i in range(wsize):
            dist.barrier()
            dist.all_gather(gather_list, grads[i].contiguous(), group=dist.group.WORLD)
            if i == wrank:
                for j in range(wsize):
                    grad_out[rows * j:rows * (j + 1)] = gather_list[j].clone()
        return grad_out


class AllGatherSlice(torch.nn.Module):
    def __init__(self):
        super(AllGatherSlice, self).__init__()

    def forward(self, input):
        return DistAllGatherSlice.apply(input)


def all_gather(tensor, memory_saving=False):
    if not memory_saving:
        return AllGather()(tensor)
    else:
        return AllGatherSlice()(tensor)


def all_reduce(tensor):
    tmp = AllGather()(tensor)
    tmp = torch.stack(tmp)
    tmp = tmp.sum(dim=0)
    return tmp
