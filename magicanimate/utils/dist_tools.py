# Copyright 2023 ByteDance and/or its affiliates.
#
# Copyright (2023) MagicAnimate Authors
#
# ByteDance, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from ByteDance or
# its affiliates is strictly prohibited.
import os
import socket
import warnings
import torch
from torch import distributed as dist


def distributed_init(args):

    if dist.is_initialized():
        warnings.warn("Distributed is already initialized, cannot initialize twice!")
        args.rank = dist.get_rank()
    else:
        print(
            f"Distributed Init (Rank {args.rank}): "
            f"{args.init_method}"
        )
        dist.init_process_group(
            backend='nccl',
            init_method=args.init_method,
            world_size=args.world_size,
            rank=args.rank,
        )
        print(
            f"Initialized Host {socket.gethostname()} as Rank "
            f"{args.rank}"
        )

        if "MASTER_ADDR" not in os.environ or "MASTER_PORT" not in os.environ:
            # Set for onboxdataloader support
            split = args.init_method.split("//")
            assert len(split) == 2, (
                "host url for distributed should be split by '//' "
                + "into exactly two elements"
            )

            split = split[1].split(":")
            assert (
                len(split) == 2
            ), "host url should be of the form <host_url>:<host_port>"
            os.environ["MASTER_ADDR"] = split[0]
            os.environ["MASTER_PORT"] = split[1]

        # perform a dummy all-reduce to initialize the NCCL communicator
        dist.all_reduce(torch.zeros(1).cuda())

        suppress_output(is_master())
        args.rank = dist.get_rank()
    return args.rank


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_nccl_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_master():
    return get_rank() == 0


def synchronize():
    if dist.is_initialized():
        dist.barrier()


def suppress_output(is_master):
    """Suppress printing on the current device. Force printing with `force=True`."""
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

    import warnings

    builtin_warn = warnings.warn

    def warn(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_warn(*args, **kwargs)

    # Log warnings only once
    warnings.warn = warn
    warnings.simplefilter("once", UserWarning)