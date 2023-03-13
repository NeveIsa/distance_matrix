exp:
	@xonsh exp.xsh

deps:
	pip install -r req.txt

data:
	python 0datagen.py --n_nodes=100 -gSTD=0.3 --nlosPERCENT=10 --nlosMAX=10 --maskmissPERCENT=50 --outputdir=distancedata

black:
	black *.py
	
cleandata:
	rm -rf distancedata/*

push:
	git add .
	git commit -m updates
	git push
