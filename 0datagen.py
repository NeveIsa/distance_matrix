import numpy as np
from fire import Fire
from pathlib import Path
from loguru import logger
import yaml


def generate(n_nodes, gSTD, nlosMAX, outputdir):
    outputdir = Path(outputdir) / f"nodes{n_nodes}g{gSTD}nlos{nlosMAX}"
    if not outputdir.exists():
        logger.info(f"creating {outputdir}")
        outputdir.mkdir()
    else:
        logger.debug(f"skipping {outputdir}")

    X = np.random.uniform(low=0, high=5, size=(n_nodes, 2))
    D = np.zeros((n_nodes, n_nodes))
    NLOS_D = np.zeros((n_nodes, n_nodes))

    for i in range(n_nodes):
        for j in range(n_nodes):
            if i < j:
                D[i, j] = D[j, i] = np.linalg.norm(X[i, :] - X[j, :])
                nlosnoise = (
                    np.random.uniform(low=0, high=nlosMAX)
                    if np.random.rand() > 0.1
                    else 0
                )
                NLOS_D[i, j] = NLOS_D[j, i] = nlosnoise + D[i, j]

    gnoise = np.random.normal(loc=0, scale=gSTD, size=D.shape)
    noisyD = NLOS_D + gnoise

    np.save(outputdir / "D", D)
    np.save(outputdir / "X", X)
    np.save(outputdir / "nlos_D", NLOS_D)
    np.save(outputdir / "noisyD", noisyD)

    info = {"N_NODES": n_nodes, "N_ANCHORS": 100}

    yaml.dump(info, open(outputdir / "info.yml", "w"))


if __name__ == "__main__":
    Fire(generate)
