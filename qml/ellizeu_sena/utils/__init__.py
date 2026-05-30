from .detect_environment import (
    has_slurm,
)

from .get_partition_status import (
    get_partition_status,
)

__all__ = [
    "has_slurm",
    "get_partition_status",
]