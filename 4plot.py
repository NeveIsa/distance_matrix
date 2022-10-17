import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from fire import Fire
from pathlib import Path
from loguru import logger


def RMSE(X, Y):
    return np.linalg.norm(X - Y)


def main(inputdir):
    inputdir = Path(inputdir)

    groundtruth = np.load(inputdir / "X.npy")
    predicted = np.load(inputdir / "3alignedX.npy")

    error = RMSE(groundtruth, predicted)
    error = round(error, 5)
    sns.scatterplot(
        x=groundtruth[:, 0], y=groundtruth[:, 1], label="gt", marker="+", alpha=1
    )
    sns.scatterplot(
        x=predicted[:, 0], y=predicted[:, 1], label="pred", markers="o", alpha=0.3
    )
    plt.title(f"rmse: {error}")

    outfile = inputdir / "4compare.png"
    plt.savefig(outfile)

    logger.info(f"plotted {outfile}")


if __name__ == "__main__":
    Fire(main)
