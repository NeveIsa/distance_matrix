exp:
	@xonsh exp.xsh

deps:
	pip -q install -r req.txt

data:
	python 0datagen.py 100 0.3 10 distancedata

black:
	black *.py
	
cleandata:
	rm -rf distancedata/*

push:
	git add .
	git commit -m updates
	git push
