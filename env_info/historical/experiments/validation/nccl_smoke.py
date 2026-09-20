import os
import datetime
import torch
import torch.distributed as dist
rank=int(os.environ['LOCAL_RANK'])
torch.cuda.set_device(rank)
dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=90))
x=torch.tensor([rank+1.0],device='cuda')
dist.all_reduce(x)
assert x.item() == 15.0, x.item()
dist.barrier()
dist.destroy_process_group()
if rank == 0: print('NCCL five-GPU all_reduce passed')
