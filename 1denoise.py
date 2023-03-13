import numpy as np
from scipy.sparse.linalg import svds
from fire import Fire
from pathlib import Path
from loguru import logger
import tensorly.decomposition as td
import tensorly as tl
from tqdm import tqdm
 
def lowrank(matrix, mask, rank=4):
    # u, s, vT = svds(matrix, k=rank)
    # reduced = u @ np.diag(s) @ vT

    w,f = td.parafac(matrix, rank=rank)
    reduced = tl.cp_to_tensor((w,f))
    return reduced
    

def sparsenoise_plus_lowrank(matrix, mask=None, rank=4, sparsity=100):
    # als
    # matrix = lowrank(UV) + sparse\
    sparse = np.zeros(matrix.shape)

    #sparse and UV
    relerr = np.inf

    pbar = tqdm(range(100))
    for p in pbar:

        # low rank estimation  given  sparse
        w,f = td.parafac(matrix - sparse, rank=rank)
        lrmatrix =  tl.cp_to_tensor((w,f)) #low rank matrix

        # sparse estimaaation given low rank
        sparse = matrix - lrmatrix
        splist = sorted(sparse.flatten(), reverse=True)
        cutoff = splist[sparsity]
        sparse[sparse<cutoff] = 0

        recmatrix = lrmatrix + sparse 
        relerr = tl.norm(recmatrix)/tl.norm(matrix)
        pbar.set_postfix({'relerr':relerr})

    return lrmatrix,sparse

def main(inputdir, mode="lowrank"):
    inputdir = Path(inputdir)
    noisymatrix = np.load(inputdir / "noisyD.npy")
    mask = np.load(inputdir / "mask.npy")
    noisymatrix **= 2  # square it
    noisymatrix *= mask
    
    if mode == "lowrank":
        denoisedmatrix = lowrank(noisymatrix, mask=mask)

    elif mode == "sparsenoise_plus_lowrank":
        denoisedmatrix,sparse = sparsenoise_plus_lowrank(noisymatrix, mask=mask)
        
    denoisedmatrix[denoisedmatrix < 0] = 0  # project to all positive
    denoisedmatrix **= 0.5  # sqrt it

    outfile = Path(inputdir) / "1denoisedD"
    np.save(outfile, denoisedmatrix)
    logger.info(f"Saved {outfile}")


if __name__ == "__main__":
    Fire(main)
