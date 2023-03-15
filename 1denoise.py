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
    

def sparse_plus_lowrank(matrix, mask=None, rank=4, sparsity=10):
    # als
    # matrix = lowrank(UV) + sparse
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

def sparse_plus_lowrank2(matrix, rank=4, sparsity=100, lmbda=1, mu=1):
    lasterr = np.inf
    ystar = np.zeros_like(matrix)
    pbar = tqdm(range(1000))
    for i in pbar:
        # low rank subproblem
        dbar = matrix - ystar
        U,S,Vt = svds(dbar, k=rank)
        lrmatrix = U@np.diag(S)@Vt
        xstar = lrmatrix/(lmbda+1)

        # sparse subproblem
        dtilde = matrix - xstar
        cutoff = sorted(dtilde.flatten(), reverse=True)[sparsity]
        sstar = (dtilde>cutoff)*1
        ystar = sstar*dtilde/(1+mu)

        err = tl.norm(matrix - xstar - ystar)**2 + lmbda*tl.norm(xstar)**2 + mu*tl.norm(ystar)**2
        relerr = (lasterr - err)/err
        lasterr = err
        pbar.set_postfix({'relerr':relerr})
        if err>0 and relerr<1e-9: break
    
    
    return lrmatrix, ystar

def main(inputdir, algo="sparse_plus_lowrank"):
    inputdir = Path(inputdir)
    noisymatrix = np.load(inputdir / "noisyD.npy")
    mask = np.load(inputdir / "mask.npy")
    noisymatrix **= 2  # square it
    noisymatrix *= mask
    
    if algo == "lowrank":
        denoisedmatrix = lowrank(noisymatrix, mask=mask)

    elif algo == "sparse_plus_lowrank":
        invmask = 1 - mask
        noisymatrix[invmask] = 10 # set the unkown to a high value
        denoisedmatrix,sparse = sparse_plus_lowrank2(noisymatrix, sparsity=50)
        # print(denoisedmatrix)
        
    denoisedmatrix[denoisedmatrix < 0] = 0  # project to all positive
    denoisedmatrix **= 0.5  # sqrt it

    outfile = Path(inputdir) / "1denoisedD"
    np.save(outfile, denoisedmatrix)
    logger.info(f"Saved {outfile}")


if __name__ == "__main__":
    Fire(main)
