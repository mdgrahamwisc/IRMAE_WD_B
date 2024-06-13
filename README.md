# IRMAE-WD-B
This repo contains a demo for Implicit Rank Minimizing Autoencoder with Weight-Decay and branches with different numbers of linear layers (IRMAE-WD-B) implemented in pytorch applied to the KSE, L=22 dataset.

-----------------------------------------------

**IRMAEWDB_AE.pt**  

Pytorch model trained for 200 epochs

-----------------------------------------------

**IRMAE_WD_B.py**

Pytorch implementation of IRMAE-WD-B applied to KSE, L=22 dataset

-----------------------------------------------

**KSE_L22.mat**

KSE, L=22 dataset

-----------------------------------------------

**Out.txt**

Output of training log

-----------------------------------------------

**TrainCurve.png**

Training log

-----------------------------------------------

**code_id_list.p**

Output of branch path selected in epoch

-----------------------------------------------

**code_musigma.p**

Mean and standard deviation of learned code space

-----------------------------------------------

**code_sValues.png**

Singular value spectra of the covariance of latent space z (this is a demo for 200 epochs, increase epochs to 1000 for sharper drop)

-----------------------------------------------

**code_svd.p**

SVD matrices U, S, V of the covariance of the learned latent space

-----------------------------------------------

**err.p**

Training log data

-----------------------------------------------

**training_svd.p**
Singular value spectra of the covariance of z as a function of training epochs

-----------------------------------------------
