import os
import sys
from enum import Enum

import torch


def is_available() -> bool:
    """
    Returns ``True`` if the distributed package is available. Otherwise,
    ``torch.distributed`` does not expose any other APIs. Currently,
    ``torch.distributed`` is available on Linux, MacOS and Windows. Set
    ``USE_DISTRIBUTED=1`` to enable it when building PyTorch from source.
    Currently, the default value is ``USE_DISTRIBUTED=1`` for Linux and Windows,
    ``USE_DISTRIBUTED=0`` for MacOS.
    """
    return hasattr(torch._C, "_c10d_init")


if is_available() and not torch._C._c10d_init():
    raise RuntimeError("Failed to initialize torch.distributed")

# Custom Runtime Errors thrown from the distributed package
DistBackendError = torch._C._DistBackendError

if is_available():
    from torch._C._distributed_c10d import _DEFAULT_FIRST_BUCKET_BYTES
    from torch._C._distributed_c10d import Backend as _Backend
    from torch._C._distributed_c10d import (BuiltinCommHookType, DebugLevel,
                                            FileStore, GradBucket, Logger,
                                            PrefixStore, ProcessGroup, Reducer,
                                            Store, TCPStore)
    from torch._C._distributed_c10d import Work as _Work
    from torch._C._distributed_c10d import (_broadcast_coalesced,
                                            _compute_bucket_assignment_by_size,
                                            _make_nccl_premul_sum,
                                            _register_builtin_comm_hook,
                                            _register_comm_hook,
                                            _test_python_store,
                                            _verify_params_across_processes,
                                            get_debug_level, set_debug_level,
                                            set_debug_level_from_env)

    if sys.platform != "win32":
        from torch._C._distributed_c10d import (HashStore,
                                                _round_robin_process_groups)

    # Variables prefixed with underscore are not auto imported
    # See the comment in `distributed_c10d.py` above `_backend` on why we expose
    # this.
    from .distributed_c10d import *  # noqa: F403
    from .distributed_c10d import (GroupMember, _all_gather_base, _backend,
                                   _c10d_error_logger,
                                   _create_process_group_wrapper, _group_count,
                                   _pg_backend_config, _pg_group_ranks,
                                   _pg_map, _pg_names, _rank_not_in_group,
                                   _reduce_scatter_base, _world, _World)
    from .remote_device import _remote_device
    from .rendezvous import (_create_store_from_options,
                             register_rendezvous_handler, rendezvous)

    set_debug_level_from_env()

else:
    # This stub is sufficient to get
    #   python test/test_public_bindings.py -k test_correct_module_names
    # working even when USE_DISTRIBUTED=0.  Feel free to add more
    # stubs as necessary.
    # We cannot define stubs directly because they confuse pyre

    class _ProcessGroupStub:
        pass

    sys.modules["torch.distributed"].ProcessGroup = _ProcessGroupStub  # type: ignore[attr-defined]
