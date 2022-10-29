import numpy as np
from numpy.linalg import norm
from loguru import logger
from pathlib import Path
from fire import Fire

def getdiff(matAfile, matBfile, normord='fro'):
    # normord = 'fro' or '2'    
    # assert normord in [2, 'fro', '2']

    matAfile = Path(matAfile)
    matBfile = Path(matBfile)

    if not matAfile.is_file():
        logger.error(f"{matAfile} not found !!!")
    else:
        A = np.load(matAfile)
        
    if not matBfile.is_file():
        logger.error(f"{matBfile} not found !!!")
    else:
        B = np.load(matBfile)
   
    diff = norm(A-B, ord=normord)/A.shape[0]
    print(diff)
    
if __name__ == "__main__":
    Fire(getdiff)
