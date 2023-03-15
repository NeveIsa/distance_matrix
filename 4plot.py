import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fire import Fire
from pathlib import Path
from loguru import logger
import yaml

def RMSE(X, Y):
    return np.linalg.norm(X - Y) / X.shape[0]


def main(inputdir):
    inputdir = Path(inputdir)

    groundtruth = np.load(inputdir / "X.npy")
    predicted = np.load(inputdir / "3alignedX.npy")

    error = RMSE(groundtruth, predicted)
    error = round(error, 5)
    plt.scatter(
        x=groundtruth[:, 0], y=groundtruth[:, 1], label="gt", marker=".",  color='orange', alpha=1, 
    )
    plt.scatter(
        x=predicted[:, 0], y=predicted[:, 1], label="pred", marker="o", facecolors='none', color='green', alpha=1,
    )
    plt.title(f"rmse: {error}")

    outfile = inputdir / "4compare.png"

    info = yaml.safe_load(open(inputdir/"info.yml").read())
    plt.title(info)
    plt.savefig(outfile)

    logger.info(f"plotted {outfile}")


if __name__ == "__main__":
    Fire(main)
