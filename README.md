# Experiments with Adversarial Autoencoders in Keras

The experiments are done within Jupyter notebooks. The notebooks are pieces of Python code with markdown texts as 
commentary. All remarks are welcome. 

# Installation 

You need to install Tensorflow, Keras, Python, SciPy/NumPy, Jupyter. Use e.g. pip3 to make installation a little bit less
of a pain (but still).

	sudo -H pip3 install tensorflow
	sudo -H pip3 install keras
	sudo -H pip3 install --upgrade --force-reinstall scipy
	sudo -H pip3 install jupyter

Check your installation by running python3 and try importing it:

	python3
	import tensorflow as tf
	import keras.backend as K

Check pip3 if you actually installed it in the "right way":

	sudo -H pip3 list | grep -i keras

# Copyright

* Author: Anne van Rossum
* Copyright: Anne van Rossum
* Date: July 12, 2018
* License: LGPLv3+, Apache Licensen 2.0, and/or MIT (triple-licensed)
