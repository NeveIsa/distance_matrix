import numpy as np
from scipy.linalg import orthogonal_procrustes
from fire import Fire
import yaml
from loguru import logger

# from scipy.spatial.transform import Rotation
from pathlib import Path


def align(inputdir):
    inputdir = Path(inputdir)

    info = yaml.safe_load(open(inputdir / "info.yml"))
    N_ANCHORS = info["N_ANCHORS"]
    # solver = Rotation.align_vectors
    solver = orthogonal_procrustes

    A = np.load(inputdir / "X.npy")
    B = np.load(inputdir / "2recoveredX.npy")

    meanA = A.mean(axis=0)
    meanB = B.mean(axis=0)

    cA = A - np.outer([1] * A.shape[0], meanA)
    cB = B - np.outer([1] * B.shape[0], meanB)

    R, scale = solver(cA, cB)  # finds R such that R takes B to A
    aligned_cB = (R @ cB.T).T

    alignedB = aligned_cB + meanA

    outfile = inputdir / "3alignedX"
    np.save(outfile, alignedB)
    logger.info(f"Saved {outfile}")


if __name__ == "__main__":
    Fire(align)
