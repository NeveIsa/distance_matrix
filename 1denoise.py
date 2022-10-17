import numpy as np
from scipy.sparse.linalg import svds
from fire import Fire
from pathlib import Path
from loguru import logger


def lowrank(matrix):
    u, s, vT = svds(matrix, k=4)
    reduced = u @ np.diag(s) @ vT
    return reduced


def sparsenoise_plus_lowrank(matrix):
    pass


def main(inputdir, mode="lowrank"):
    inputdir = Path(inputdir)
    noisymatrix = np.load(inputdir / "noisyD.npy")
    noisymatrix **= 2  # square it

    if mode == "lowrank":
        denoisedmatrix = lowrank(noisymatrix)

    elif mode == "sparsenoise_plus_lowrank":
        denoisedmatrix = sparsenoise_plus_lowrank(noisymatrix)

    denoisedmatrix[denoisedmatrix < 0] = 0  # project to all positive
    denoisedmatrix **= 0.5  # sqrt it

    outfile = Path(inputdir) / "1denoisedD"
    np.save(outfile, denoisedmatrix)
    logger.info(f"Saved {outfile}")


if __name__ == "__main__":
    Fire(main)
