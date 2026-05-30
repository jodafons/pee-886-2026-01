# qml/ellizeu_sena/utils/get_partition_status.py

import subprocess


def get_partition_status(partition_name):
    """
    Return statistics for a SLURM partition.

    Parameters
    ----------
    partition_name : str

    Returns
    -------
    dict
    """

    output = subprocess.check_output(
        ["sinfo", "-N", "-h"],
        text=True,
    )

    total = 0
    idle = 0
    mixed = 0
    allocated = 0
    unavailable = 0

    for line in output.splitlines():

        columns = line.split()

        if len(columns) < 4:
            continue

        partition = columns[1]
        state = columns[2].lower()

        partition = partition.replace("*", "")

        if partition != partition_name:
            continue

        total += 1

        if state.startswith("idle"):
            idle += 1

        elif state.startswith("mix"):
            mixed += 1

        elif state.startswith("alloc"):
            allocated += 1

        else:
            unavailable += 1

    return {
        "partition": partition_name,
        "total_nodes": total,
        "idle_nodes": idle,
        "mixed_nodes": mixed,
        "allocated_nodes": allocated,
        "unavailable_nodes": unavailable,
        "available_nodes": idle + mixed,
    }