import numpy as np
from scipy.sparse.linalg import svds
from fire import Fire
from pathlib import Path
from loguru import logger
import tensorly.decomposition as td

def lowrank(matrix, mask, rank=4):
    u, s, vT = svds(matrix, k=rank)
    reduced = u @ np.diag(s) @ vT
    return reduced
    

def sparsenoise_plus_lowrank(matrix, mask=None, rank=4, sparsity=100):
    # als
    # matrix = lowrank(UV) + sparse\
    m,n = matrix.shape
    U = np.random.rand(m,rank)
    V = np.random.rand(rank,n)
    sparse = np.zeros((m,n))

    #fix sparse and U
    lrmatrix = matrix - sparse #low rank matrix
    (w,f) = td.non_negative_parafac(lrmatrix, rank=rank)
    U,V = f
    print(U.shape, V.shape)
    exit()


def main(inputdir, mode="sparsenoise_plus_lowrank"):
    inputdir = Path(inputdir)
    noisymatrix = np.load(inputdir / "noisyD.npy")
    mask = np.load(inputdir / "mask.npy")
    noisymatrix **= 2  # square it

    if mode == "lowrank":
        denoisedmatrix = lowrank(noisymatrix, mask=mask)

    elif mode == "sparsenoise_plus_lowrank":
        denoisedmatrix = sparsenoise_plus_lowrank(noisymatrix, mask=mask)
        
    denoisedmatrix[denoisedmatrix < 0] = 0  # project to all positive
    denoisedmatrix **= 0.5  # sqrt it

    outfile = Path(inputdir) / "1denoisedD"
    np.save(outfile, denoisedmatrix)
    logger.info(f"Saved {outfile}")


if __name__ == "__main__":
    Fire(main)
