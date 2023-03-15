from fire import Fire
import numpy as np
from pathlib import Path
from loguru import logger
from scipy.sparse.linalg import svds


def distance2gram(inputdir):
    inputdir = Path(inputdir)

    D = np.load(inputdir / "1denoisedD.npy")
    X = np.zeros(D.shape)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X[i, j] = 0.5 * (D[0, i] ** 2 + D[j, 0] ** 2 - D[i, j] ** 2)

    u, s, vT = svds(X, k=2)
    recoveredX = u @ np.diag(np.sqrt(s))

    outfile = inputdir / "2recoveredX"
    np.save(outfile, recoveredX)
    logger.info(f"Saved {outfile}")

def lle(inputdir, k=10):
    """
        k = k in k-nearest neighbours
    """
    inputdir = Path(inputdir)
    
    D = np.load(inputdir / "1denoisedD.npy")
    X = np.zeros(D.shape)
       
    for i in range(D.shape[0]):
        knn = np.argsort(D[i,:])[:k]
        if not i in knn: 
            knn = np.array(knn.tolist()+[i])
            # d =

def main(inputdir, algo='mds'):
    if algo=='mds':
        distance2gram(inputdir)
    elif algo=='lle':
        lle(inputdir)
    
if __name__ == "__main__":
    Fire(main)
