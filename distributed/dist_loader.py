import atexit
import logging
import os
from typing import Any, Optional, Union

import torch.distributed
import torch.multiprocessing as mp

from torch_geometric.distributed import DistNeighborSampler
from torch_geometric.distributed.dist_context import DistContext
from torch_geometric.loader.base import DataLoaderIterator


class DistLoader:
    r"""A base class for creating distributed data loading routines.

    Args:
        current_ctx (DistContext): Distributed context info of the current
            process.
        channel (mp.Queue, optional): A communication channel for messages.
            (default: :obj:`None`)
    """
    def __init__(
        self,
        current_ctx: DistContext,
        channel: Optional[mp.Queue] = None,
        dist_sampler: DistNeighborSampler = None,
        **kwargs,
    ):
        self.dist_sampler = dist_sampler
        self.current_ctx = current_ctx
        self.channel = channel
        self.pid = mp.current_process().pid
        self.num_workers = kwargs.get('num_workers', 0)

        logging.info(f"[{self}] Initialized with PID={self.pid}")

    def channel_get(self, out: Any) -> Any:
        if self.channel:
            out = self.channel.get()
            logging.debug(f"[{self}] Retrieved message")
        return out

    def reset_channel(self, channel=None):
        # clean remaining queue items and restart new queue
        logging.debug(f'{self} Resetting msg channel')
        while not self.channel.empty():
            self.channel.get_nowait()

        torch.distributed.barrier()

        self.channel = channel or mp.Queue()
        self.dist_sampler.channel = self.channel

    def worker_init_fn(self, worker_id: int):
        try:
            num_sampler_proc = self.num_workers if self.num_workers > 0 else 1
            self.current_ctx_worker = DistContext(
                world_size=self.current_ctx.world_size * num_sampler_proc,
                rank=self.current_ctx.rank * num_sampler_proc + worker_id,
                global_world_size=self.current_ctx.world_size *
                num_sampler_proc,
                global_rank=self.current_ctx.rank * num_sampler_proc +
                worker_id,
                group_name='mp_sampling_worker',
            )

            logging.info(
                f"Worker-{worker_id} initialized "
                f"(current_ctx_worker={self.current_ctx_worker.worker_name})")
            self.dist_sampler.init_sampler_instance()
            logging.info(
                f"Sampler instance initialized in worker-{worker_id}")

        except RuntimeError:
            raise RuntimeError(f"`{self}.init_fn()` could not initialize the "
                               f"worker loop of the neighbor sampler")

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(pid={self.pid})'

    def __enter__(self) -> DataLoaderIterator:
        # fetch a single batch for init
        self._prefetch_old = self.prefetch_factor
        self.prefetch_factor = 1
        self._iterator = self._get_iterator()
        return self._iterator

    def __exit__(self, *args) -> None:
        if self.channel:
            self.reset_channel()
        if self._iterator:
            del self._iterator
            torch.distributed.barrier()
            self._iterator = None
            self.prefetch_factor = self._prefetch_old

