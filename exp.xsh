from glob import glob
from pathlib import Path

datadir = "distancedata"

def generatedata():
	python 0datagen.py 500 0.1 0 @(datadir)


def process():
	print()
	dataset = glob( str(Path(datadir)/"*")  )
	for ds in dataset:
		print(f"Processing -> {ds}")				
		python 1denoise.py @(ds)
		python 2mds.py @(ds)
		python 3align.py @(ds)
		python 4plot.py @(ds)
		feh @(ds)/4compare.png
		print()

# MAIN #
make cleandata
generatedata()
process()
