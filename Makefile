pip=ve_ionmob/bin/pip
py=ve_ionmob/bin/python

install:
	virtualenv ve_ionmob
	$(pip) install -e .
	$(pip) install IPython
py:
	$(py) -m IPython
subl:
	subl -n .
