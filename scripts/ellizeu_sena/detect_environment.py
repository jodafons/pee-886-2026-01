# scripts/ellizeu_sena/detect_environment.py

import shutil

def has_slurm():
    return shutil.which("squeue") is not None