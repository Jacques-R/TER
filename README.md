# TER
Repository for the TER of Jacques Rossel, Sacha Carniere and Reda Ouazzani


The simplest way to run our code is to use Google colabs. a link to our Notebook here #TODO


First you will need to clone this git repository to have acces to the code and also another repository with the dataset we use.

! git clone https://github.com/Jacques-R/TER.git

! git clone https://gitlab.com/baptiste.pouthier/google-speech-dataset.git

Then import all those library:
```
! pip install numpy
! pip install termcolor
! pip install scipy
! pip install sklearn
! pip install scikit-learn
! pip install tensorflow-gpu==1.6.0
! pip install keras==2.1.4
! pip2.7 install scikits.audiolab
```
then you can start extracting the features to do so you need to go to at 

TER/Speech-Command-Recognition-with-Capsule-Network/core/

 in our repository 
 and then execute the feature extraction file :
 
 ! python2.7 feature_generation.py
 
 You can then start Training and testing different Network (DNN,CNN,CRNN,RNN) with differen settings (open or close set ,type of noise , strenght of the noise)  
 
 using this command and changing the parameters:
 
! python main.py --model='CNN' --ex_name='ref_2014icassp_dnn512' --is_training='TRAIN' --model_size_info 512 512 512 

here we have a training oon the model ref_2014icassp_dnn512 (a DNN) whith no noise and a close set.
