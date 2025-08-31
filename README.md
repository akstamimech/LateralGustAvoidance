# LateralGustAvoidance
Project work for AE4350 Bio Inspired intelligence For Aerospace Engineering.


In order to run this, you NEED to have Citation_controller.dat in your directory. The closed loop matrices are stored in this file and they come from MATLAB - coalition.m .
Blobtrainer.py uses the environment citation_env_gauss.py. Training parameters are inside blobtrainer.m. This will save model and normalization parameters in a directory of your choice. 
These params can then be used to deploy a demo in demomodelgauss.py.

The other files serve the same purpose, except for Sinusoidal distribution instead of gaussian. Training can take a long time (>1 million steps) if you would like to see any substantial changes. 
