from glob import glob
from pathlib import Path

datadir = "distancedata"

def generatedata():
	python 0datagen.py --n_nodes=500 --gSTD=0.1 --nlosPERCENT=10 --nlosMAX=0 --maskmissPERCENT=20 --outputdir=@(datadir)


def process():
	print()
	dataset = glob( str(Path(datadir)/"*")  )
	for ds in dataset:
		print(f"Processing -> {ds}")				
		python 1denoise.py --inputdir=@(ds)
		# exit(0)
		python 2mds.py @(ds)
		python 3align.py @(ds)
		python 4plot.py @(ds)
		feh @(ds)/4compare.png
		print()

# MAIN #
make cleandata
generatedata()
process()
