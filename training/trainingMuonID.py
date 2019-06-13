#!/usr/bin/env python

from ROOT import TMVA, TFile, TTree, TCut
from subprocess import call
from os.path import isfile
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import keras
import h5py

from keras.models import Sequential
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint

##### FUNCTIONS

def getKerasModel(inputDim, modelName, nLayers = 3, layerSize = 200, dropValue = 0.2, optLabel = 'adam'):
    model = Sequential()
    model.add(Dense(layerSize, activation='relu', kernel_initializer='normal', input_dim=inputDim))
    if dropValue != 0:
            model.add(Dropout(dropValue))

    for i in range(1, nLayers):
        model.add(Dense(layerSize, activation='relu', kernel_initializer='normal'))
        if dropValue != 0:
            model.add(Dropout(dropValue))

    model.add(Dense(2, activation='softmax'))

    opt = Adam(lr=0.001)
    if optLabel == 'sgd':
        opt = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.save(modelName)
    model.summary()
    return

##### MAIN

# Setup TMVA
TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()

nLayers = sys.argv[1]
layerSize = sys.argv[2]
dropValue = sys.argv[3]

nLayers = int(nLayers)
layerSize = int(layerSize)
dropValue = float(dropValue)

isoflag = 'wIso'

if isoflag != 'wIso' and isoflag != 'woIso':
    print 'Invalid argument isoflag'
    sys.exit()


outputName = 'TMVAMuonID.root'
output = TFile.Open(outputName, 'RECREATE')

# Factory
factory = TMVA.Factory('TMVAClassification', output,
                       '!V:!Silent:Color:DrawProgressBar:AnalysisType=Classification')

# Load data
#2017
dataBs17 = TFile.Open('bankBsJpsiPhi17.root')
dataBsD017 = TFile.Open('bankBsJpsiPhiDG017.root')
dataBu17 = TFile.Open('bankBuJpsiK17.root')
dataBd17 = TFile.Open('bankBdJpsiKx17.root')

treeBs17 = dataBs17.Get('PDsecondTree')
treeBsD017 = dataBsD017.Get('PDsecondTree')
treeBu17 = dataBu17.Get('PDsecondTree')
treeBd17 = dataBd17.Get('PDsecondTree')

#2018
dataBs18 = TFile.Open('bankBsJpsiPhi18.root')
dataBsD018 = TFile.Open('bankBsJpsiPhiDG018.root')
dataBu18 = TFile.Open('bankBuJpsiK18.root')
dataBd18 = TFile.Open('bankBdJpsiKx18.root')

treeBs18 = dataBs18.Get('PDsecondTree')
treeBsD018 = dataBsD018.Get('PDsecondTree')
treeBu18 = dataBu18.Get('PDsecondTree')
treeBd18 = dataBd18.Get('PDsecondTree')

dataloader = TMVA.DataLoader('dataset')

# add variables
varList = [
    ('muoPt', 'F')
    ,('muoEta', 'F')
    ,('muoSegmComp', 'F')
    ,('muoChi2LM', 'F')
    ,('muoChi2LP', 'F')
    ,('muoGlbTrackTailProb', 'F')
    ,('muoIValFrac', 'F')
    ,('muoLWH', 'I')
    ,('muoTrkKink', 'F')
    ,('muoGlbKinkFinderLOG', 'F')
    ,('muoTimeAtIpInOutErr', 'F')
    ,('muoOuterChi2', 'F')
    ,('muoInnerChi2', 'F')
    ,('muoTrkRelChi2', 'F')
    ,('muoVMuonHitComb', 'I')
    ,('muoGlbDeltaEtaPhi', 'F')
    ,('muoStaRelChi2', 'F')
    ,('muoTimeAtIpInOut', 'F')
    ,('muoValPixHits', 'I')
    ,('muoNTrkVHits', 'I')
    ,('muoGNchi2', 'F')
    ,('muoVMuHits', 'I')
    ,('muoNumMatches', 'F')
    ,('muoQprod', 'I')
    ]

if isoflag == 'wIso':
    varList.append(('muoPFiso', 'F'))

nVars = 0
for var in varList:
    dataloader.AddVariable( var[0], var[1] )
    nVars += 1

dataloader.AddSpectator( 'muoEvt', 'I' )

# prepare dataloader

mycutgen = '1'
#mycutgen = '&&(muoEvt%10)<8'

mycuts = mycutgen + '&&abs(muoLund)==13&&muoAncestor>=0'
mycutb = mycutgen + '&&abs(muoLund)!=13'

sgnW = 0.1
bkgW = 1.0

dataloader.AddSignalTree(treeBs17, sgnW)
dataloader.AddSignalTree(treeBsD017, sgnW)
dataloader.AddSignalTree(treeBu17, sgnW)
dataloader.AddSignalTree(treeBd17, sgnW)

dataloader.AddBackgroundTree(treeBs17, bkgW)
dataloader.AddBackgroundTree(treeBsD017, bkgW)
dataloader.AddBackgroundTree(treeBu17, bkgW)
dataloader.AddBackgroundTree(treeBd17, bkgW)

dataloader.AddSignalTree(treeBs18, sgnW)
dataloader.AddSignalTree(treeBsD018, sgnW)
dataloader.AddSignalTree(treeBu18, sgnW)
dataloader.AddSignalTree(treeBd18, sgnW)

dataloader.AddBackgroundTree(treeBs18, bkgW)
dataloader.AddBackgroundTree(treeBsD018, bkgW)
dataloader.AddBackgroundTree(treeBu18, bkgW)
dataloader.AddBackgroundTree(treeBd18, bkgW)

##### 2017
# Sgn = 2577094
# Bkg =  162201
#
##### 2018
# Sgn = 4049212
# Bkg =  289772
#
##### Tot
# Sgn = 6626306
# Bkg =  451973

# Tr:Va:Te 50%:25%:25%
nBkg = '338980'
nSgn = '4969730'
nBkgTest = '0'
nSgnTest = '0'


dataloaderOpt = 'nTrain_Signal=' + nSgn + ':nTrain_Background=' + nBkg + ':nTest_Signal=' + nSgnTest + ':nTest_Background=' + nBkgTest
dataloaderOpt += ':SplitMode=Random:NormMode=EqualNumEvents:!V'

dataloader.PrepareTrainingAndTestTree(TCut(mycuts), TCut(mycutb), dataloaderOpt)

# Define model
modelName = 'modelMuonID'
if isoflag == 'woIso':
    modelName += isoflag

modelName += '.h5'

getKerasModel(nVars, modelName, nLayers, layerSize, dropValue)

# Book methods
dnnOptions = '!H:!V:FilenameModel=' + modelName + ':NumEpochs=40:TriesEarlyStopping=10:BatchSize=1024:ValidationSize=33%:Tensorboard=./logs'

iVar = 0
preprocessingOptions = ':VarTransform=N'
preprocessingOptions += ',G('
for var in varList:
    if var[0] == 'muoQprod':
        iVar += 1
        continue
    preprocessingOptions += '_V' + str(iVar) + '_' + ','
    iVar += 1
preprocessingOptions = preprocessingOptions[:-1]
preprocessingOptions +=  '),N'

dnnName = 'DNNMuonID'
if isoflag == 'woIso':
    dnnName += isoflag

factory.BookMethod(dataloader, TMVA.Types.kPyKeras, dnnName, dnnOptions + preprocessingOptions)


# Run training, test and evaluation
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()

#TMVA.TMVAGui("TMVA_keras.root")





# DataSetInfo              : [dataset] : Added class "Signal"
#                          : Add Tree PDsecondTree of type Signal with 1479924 events
#                          : Add Tree PDsecondTree of type Signal with 20302 events
#                          : Add Tree PDsecondTree of type Signal with 387746 events
#                          : Add Tree PDsecondTree of type Signal with 1007016 events
# DataSetInfo              : [dataset] : Added class "Background"
#                          : Add Tree PDsecondTree of type Background with 1479924 events
#                          : Add Tree PDsecondTree of type Background with 20302 events
#                          : Add Tree PDsecondTree of type Background with 387746 events
#                          : Add Tree PDsecondTree of type Background with 1007016 events
#                          : Add Tree PDsecondTree of type Signal with 1057895 events
#                          : Add Tree PDsecondTree of type Signal with 2316472 events
#                          : Add Tree PDsecondTree of type Signal with 389846 events
#                          : Add Tree PDsecondTree of type Signal with 820482 events
#                          : Add Tree PDsecondTree of type Background with 1057895 events
#                          : Add Tree PDsecondTree of type Background with 2316472 events
#                          : Add Tree PDsecondTree of type Background with 389846 events
#                          : Add Tree PDsecondTree of type Background with 820482 events
# WARNING:tensorflow:From /home/alberto/.local/lib/python2.7/site-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
# Instructions for updating:
# Colocations handled automatically by placer.
# WARNING:tensorflow:From /usr/local/lib/python2.7/dist-packages/keras/backend/tensorflow_backend.py:3135: calling dropout (from tensorflow.python.ops.nn_ops) with keep_prob is deprecated and will be removed in a future version.
# Instructions for updating:
# Please use `rate` instead of `keep_prob`. Rate should be set to `rate = 1 - keep_prob`.
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# dense_1 (Dense)              (None, 200)               5200      
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 200)               0         
# _________________________________________________________________
# dense_2 (Dense)              (None, 200)               40200     
# _________________________________________________________________
# dropout_2 (Dropout)          (None, 200)               0         
# _________________________________________________________________
# dense_3 (Dense)              (None, 2)                 402       
# =================================================================
# Total params: 45,802
# Trainable params: 45,802
# Non-trainable params: 0
# _________________________________________________________________
# Factory                  : Booking method: DNNMuonID
#                          : 
# DNNMuonID                : [dataset] : Create Transformation "N" with events from all classes.
#                          : 
#                          : Transformation, Variable selection : 
#                          : Input : variable 'muoPt' <---> Output : variable 'muoPt'
#                          : Input : variable 'muoEta' <---> Output : variable 'muoEta'
#                          : Input : variable 'muoSegmComp' <---> Output : variable 'muoSegmComp'
#                          : Input : variable 'muoChi2LM' <---> Output : variable 'muoChi2LM'
#                          : Input : variable 'muoChi2LP' <---> Output : variable 'muoChi2LP'
#                          : Input : variable 'muoGlbTrackTailProb' <---> Output : variable 'muoGlbTrackTailProb'
#                          : Input : variable 'muoIValFrac' <---> Output : variable 'muoIValFrac'
#                          : Input : variable 'muoLWH' <---> Output : variable 'muoLWH'
#                          : Input : variable 'muoTrkKink' <---> Output : variable 'muoTrkKink'
#                          : Input : variable 'muoGlbKinkFinderLOG' <---> Output : variable 'muoGlbKinkFinderLOG'
#                          : Input : variable 'muoTimeAtIpInOutErr' <---> Output : variable 'muoTimeAtIpInOutErr'
#                          : Input : variable 'muoOuterChi2' <---> Output : variable 'muoOuterChi2'
#                          : Input : variable 'muoInnerChi2' <---> Output : variable 'muoInnerChi2'
#                          : Input : variable 'muoTrkRelChi2' <---> Output : variable 'muoTrkRelChi2'
#                          : Input : variable 'muoVMuonHitComb' <---> Output : variable 'muoVMuonHitComb'
#                          : Input : variable 'muoGlbDeltaEtaPhi' <---> Output : variable 'muoGlbDeltaEtaPhi'
#                          : Input : variable 'muoStaRelChi2' <---> Output : variable 'muoStaRelChi2'
#                          : Input : variable 'muoTimeAtIpInOut' <---> Output : variable 'muoTimeAtIpInOut'
#                          : Input : variable 'muoValPixHits' <---> Output : variable 'muoValPixHits'
#                          : Input : variable 'muoNTrkVHits' <---> Output : variable 'muoNTrkVHits'
#                          : Input : variable 'muoGNchi2' <---> Output : variable 'muoGNchi2'
#                          : Input : variable 'muoVMuHits' <---> Output : variable 'muoVMuHits'
#                          : Input : variable 'muoNumMatches' <---> Output : variable 'muoNumMatches'
#                          : Input : variable 'muoQprod' <---> Output : variable 'muoQprod'
#                          : Input : variable 'muoPFiso' <---> Output : variable 'muoPFiso'
# DNNMuonID                : [dataset] : Create Transformation "G" with events from all classes.
#                          : 
#                          : Transformation, Variable selection : 
#                          : Input : variable 'muoPt' <---> Output : variable 'muoPt'
#                          : Input : variable 'muoEta' <---> Output : variable 'muoEta'
#                          : Input : variable 'muoSegmComp' <---> Output : variable 'muoSegmComp'
#                          : Input : variable 'muoChi2LM' <---> Output : variable 'muoChi2LM'
#                          : Input : variable 'muoChi2LP' <---> Output : variable 'muoChi2LP'
#                          : Input : variable 'muoGlbTrackTailProb' <---> Output : variable 'muoGlbTrackTailProb'
#                          : Input : variable 'muoIValFrac' <---> Output : variable 'muoIValFrac'
#                          : Input : variable 'muoLWH' <---> Output : variable 'muoLWH'
#                          : Input : variable 'muoTrkKink' <---> Output : variable 'muoTrkKink'
#                          : Input : variable 'muoGlbKinkFinderLOG' <---> Output : variable 'muoGlbKinkFinderLOG'
#                          : Input : variable 'muoTimeAtIpInOutErr' <---> Output : variable 'muoTimeAtIpInOutErr'
#                          : Input : variable 'muoOuterChi2' <---> Output : variable 'muoOuterChi2'
#                          : Input : variable 'muoInnerChi2' <---> Output : variable 'muoInnerChi2'
#                          : Input : variable 'muoTrkRelChi2' <---> Output : variable 'muoTrkRelChi2'
#                          : Input : variable 'muoVMuonHitComb' <---> Output : variable 'muoVMuonHitComb'
#                          : Input : variable 'muoGlbDeltaEtaPhi' <---> Output : variable 'muoGlbDeltaEtaPhi'
#                          : Input : variable 'muoStaRelChi2' <---> Output : variable 'muoStaRelChi2'
#                          : Input : variable 'muoTimeAtIpInOut' <---> Output : variable 'muoTimeAtIpInOut'
#                          : Input : variable 'muoValPixHits' <---> Output : variable 'muoValPixHits'
#                          : Input : variable 'muoNTrkVHits' <---> Output : variable 'muoNTrkVHits'
#                          : Input : variable 'muoGNchi2' <---> Output : variable 'muoGNchi2'
#                          : Input : variable 'muoVMuHits' <---> Output : variable 'muoVMuHits'
#                          : Input : variable 'muoNumMatches' <---> Output : variable 'muoNumMatches'
#                          : Input : variable 'muoPFiso' <---> Output : variable 'muoPFiso'
# DNNMuonID                : [dataset] : Create Transformation "N" with events from all classes.
#                          : 
#                          : Transformation, Variable selection : 
#                          : Input : variable 'muoPt' <---> Output : variable 'muoPt'
#                          : Input : variable 'muoEta' <---> Output : variable 'muoEta'
#                          : Input : variable 'muoSegmComp' <---> Output : variable 'muoSegmComp'
#                          : Input : variable 'muoChi2LM' <---> Output : variable 'muoChi2LM'
#                          : Input : variable 'muoChi2LP' <---> Output : variable 'muoChi2LP'
#                          : Input : variable 'muoGlbTrackTailProb' <---> Output : variable 'muoGlbTrackTailProb'
#                          : Input : variable 'muoIValFrac' <---> Output : variable 'muoIValFrac'
#                          : Input : variable 'muoLWH' <---> Output : variable 'muoLWH'
#                          : Input : variable 'muoTrkKink' <---> Output : variable 'muoTrkKink'
#                          : Input : variable 'muoGlbKinkFinderLOG' <---> Output : variable 'muoGlbKinkFinderLOG'
#                          : Input : variable 'muoTimeAtIpInOutErr' <---> Output : variable 'muoTimeAtIpInOutErr'
#                          : Input : variable 'muoOuterChi2' <---> Output : variable 'muoOuterChi2'
#                          : Input : variable 'muoInnerChi2' <---> Output : variable 'muoInnerChi2'
#                          : Input : variable 'muoTrkRelChi2' <---> Output : variable 'muoTrkRelChi2'
#                          : Input : variable 'muoVMuonHitComb' <---> Output : variable 'muoVMuonHitComb'
#                          : Input : variable 'muoGlbDeltaEtaPhi' <---> Output : variable 'muoGlbDeltaEtaPhi'
#                          : Input : variable 'muoStaRelChi2' <---> Output : variable 'muoStaRelChi2'
#                          : Input : variable 'muoTimeAtIpInOut' <---> Output : variable 'muoTimeAtIpInOut'
#                          : Input : variable 'muoValPixHits' <---> Output : variable 'muoValPixHits'
#                          : Input : variable 'muoNTrkVHits' <---> Output : variable 'muoNTrkVHits'
#                          : Input : variable 'muoGNchi2' <---> Output : variable 'muoGNchi2'
#                          : Input : variable 'muoVMuHits' <---> Output : variable 'muoVMuHits'
#                          : Input : variable 'muoNumMatches' <---> Output : variable 'muoNumMatches'
#                          : Input : variable 'muoQprod' <---> Output : variable 'muoQprod'
#                          : Input : variable 'muoPFiso' <---> Output : variable 'muoPFiso'
#                          : Load model from file: modelMuonID.h5
# Factory                  : Train all methods
# DataSetFactory           : [dataset] : Number of events in input trees
#                          : Dataset[dataset] :     Signal     requirement: "1&&abs(muoLund)==13&&muoAncestor>=0"
#                          : Dataset[dataset] :     Signal          -- number of events passed: 6626306  / sum of weights: 666225
#                          : Dataset[dataset] :     Signal          -- efficiency             : 0.84216
#                          : Dataset[dataset] :     Background requirement: "1&&abs(muoLund)!=13"
#                          : Dataset[dataset] :     Background      -- number of events passed: 451924  / sum of weights: 451924
#                          : Dataset[dataset] :     Background      -- efficiency             : 0.0592669
#                          : Dataset[dataset] :  you have opted for interpreting the requested number of training/testing events
#                          :  to be the number of events AFTER your preselection cuts
#                          : 
#                          : Dataset[dataset] :  you have opted for interpreting the requested number of training/testing events
#                          :  to be the number of events AFTER your preselection cuts
#                          : 
#                          : Dataset[dataset] : Weight renormalisation mode: "EqualNumEvents": renormalises all event classes ...
#                          : Dataset[dataset] :  such that the effective (weighted) number of events in each class is the same 
#                          : Dataset[dataset] :  (and equals the number of events (entries) given for class=0 )
#                          : Dataset[dataset] : ... i.e. such that Sum[i=1..N_j]{w_i} = N_classA, j=classA, classB, ...
#                          : Dataset[dataset] : ... (note that N_j is the sum of TRAINING events
#                          : Dataset[dataset] :  ..... Testing events are not renormalised nor included in the renormalisation factor!)
#                          : Number of training and testing events
#                          : ---------------------------------------------------------------------------
#                          : Signal     -- training events            : 4969730
#                          : Signal     -- testing events             : 1656576
#                          : Signal     -- training and testing events: 6626306
#                          : Dataset[dataset] : Signal     -- due to the preselection a scaling factor has been applied to the numbers of requested events: 0.868997
#                          : Background -- training events            : 338980
#                          : Background -- testing events             : 112944
#                          : Background -- training and testing events: 451924
#                          : Dataset[dataset] : Background -- due to the preselection a scaling factor has been applied to the numbers of requested events: 0.0592669
#                          : 
# DataSetInfo              : Correlation matrix (Signal):
#                          : --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                          :                        muoPt  muoEta muoSegmComp muoChi2LM muoChi2LP muoGlbTrackTailProb muoIValFrac  muoLWH muoTrkKink muoGlbKinkFinderLOG muoTimeAtIpInOutErr muoOuterChi2 muoInnerChi2 muoTrkRelChi2 muoVMuonHitComb muoGlbDeltaEtaPhi muoStaRelChi2 muoTimeAtIpInOut muoValPixHits muoNTrkVHits muoGNchi2 muoVMuHits muoNumMatches muoQprod muoPFiso
#                          :               muoPt:  +1.000  -0.001      +0.107    -0.034    +0.031              -0.002      -0.017  -0.125     -0.040              -0.246              -0.185       -0.084       +0.041        -0.031          +0.146            -0.314        +0.010           +0.018        -0.137       -0.159    -0.002     +0.276        +0.116   +0.057   -0.041
#                          :              muoEta:  -0.001  +1.000      +0.019    -0.001    -0.000              +0.001      -0.003  -0.006     -0.003              -0.001              -0.008       +0.001       +0.001        +0.005          +0.015            -0.003        +0.000           -0.002        -0.043       +0.000    +0.001     +0.011        +0.021   +0.002   +0.001
#                          :         muoSegmComp:  +0.107  +0.019      +1.000    -0.078    -0.120              -0.004      +0.021  +0.056     +0.001              -0.056              -0.028       -0.073       -0.009        +0.109          +0.345            -0.354        -0.003           +0.038        +0.091       +0.077    -0.002     +0.236        +0.708   +0.068   -0.031
#                          :           muoChi2LM:  -0.034  -0.001      -0.078    +1.000    +0.309              +0.097      -0.004  -0.051     -0.002              -0.073              +0.048       +0.209       +0.001        -0.017          -0.284            +0.210        +0.036           +0.015        -0.028       -0.049    +0.002     -0.248        -0.096   -0.368   +0.014
#                          :           muoChi2LP:  +0.031  -0.000      -0.120    +0.309    +1.000              +0.104      -0.010  -0.025     -0.002              +0.023              -0.016       +0.042       +0.005        -0.023          -0.073            +0.290        +0.029           +0.001        -0.031       -0.033    +0.001     -0.044        -0.074   -0.041   +0.009
#                          : muoGlbTrackTailProb:  -0.002  +0.001      -0.004    +0.097    +0.104              +1.000      -0.000  +0.006     +0.040              +0.045              +0.010       -0.002       +0.042        +0.035          +0.000            +0.022        +0.485           +0.004        +0.011       +0.011    +0.003     -0.001        +0.003   -0.016   +0.003
#                          :         muoIValFrac:  -0.017  -0.003      +0.021    -0.004    -0.010              -0.000      +1.000  +0.280     -0.004              -0.003              +0.039       -0.006       -0.030        +0.037          +0.021            -0.026        +0.001           +0.010        +0.194       +0.252    -0.003     -0.025        +0.047   -0.005   -0.013
#                          :              muoLWH:  -0.125  -0.006      +0.056    -0.051    -0.025              +0.006      +0.280  +1.000     +0.079              +0.014              +0.104       +0.005       -0.041        +0.320          +0.041            -0.108        -0.001           +0.038        +0.516       +0.865    -0.002     -0.116        +0.121   +0.059   -0.008
#                          :          muoTrkKink:  -0.040  -0.003      +0.001    -0.002    -0.002              +0.040      -0.004  +0.079     +1.000              +0.018              +0.038       +0.002       +0.683        +0.336          +0.006            -0.001        +0.000           +0.009        +0.068       +0.098    -0.000     -0.035        +0.018   -0.003   +0.002
#                          : muoGlbKinkFinderLOG:  -0.246  -0.001      -0.056    -0.073    +0.023              +0.045      -0.003  +0.014     +0.018              +1.000              +0.297       -0.268       +0.012        +0.022          +0.345            +0.061        +0.018           -0.001        +0.013       +0.013    +0.001     +0.255        +0.079   +0.075   +0.015
#                          : muoTimeAtIpInOutErr:  -0.185  -0.008      -0.028    +0.048    -0.016              +0.010      +0.039  +0.104     +0.038              +0.297              +1.000       -0.255       -0.028        +0.279          +0.024            -0.002        +0.022           +0.065        +0.197       +0.188    +0.001     -0.330        +0.073   -0.077   -0.015
#                          :        muoOuterChi2:  -0.084  +0.001      -0.073    +0.209    +0.042              -0.002      -0.006  +0.005     +0.002              -0.268              -0.255       +1.000       -0.005        -0.014          -0.345            +0.216        -0.009           +0.014        -0.024       -0.003    -0.000     -0.283        -0.153   -0.191   +0.017
#                          :        muoInnerChi2:  +0.041  +0.001      -0.009    +0.001    +0.005              +0.042      -0.030  -0.041     +0.683              +0.012              -0.028       -0.005       +1.000        +0.422          -0.001            +0.004        -0.000           -0.005        -0.048       -0.033    +0.000     +0.030        -0.019   +0.003   +0.006
#                          :       muoTrkRelChi2:  -0.031  +0.005      +0.109    -0.017    -0.023              +0.035      +0.037  +0.320     +0.336              +0.022              +0.279       -0.014       +0.422        +1.000          +0.169            -0.161        +0.006           +0.086        +0.255       +0.452    -0.001     -0.174        +0.272   -0.003   -0.028
#                          :     muoVMuonHitComb:  +0.146  +0.015      +0.345    -0.284    -0.073              +0.000      +0.021  +0.041     +0.006              +0.345              +0.024       -0.345       -0.001        +0.169          +1.000            -0.444        -0.006           +0.063        +0.093       +0.068    -0.004     +0.758        +0.615   +0.257   -0.036
#                          :   muoGlbDeltaEtaPhi:  -0.314  -0.003      -0.354    +0.210    +0.290              +0.022      -0.026  -0.108     -0.001              +0.061              -0.002       +0.216       +0.004        -0.161          -0.444            +1.000        +0.001           -0.059        -0.101       -0.126    +0.005     -0.343        -0.425   -0.168   +0.043
#                          :       muoStaRelChi2:  +0.010  +0.000      -0.003    +0.036    +0.029              +0.485      +0.001  -0.001     +0.000              +0.018              +0.022       -0.009       -0.000        +0.006          -0.006            +0.001        +1.000           +0.000        +0.003       +0.001    +0.001     -0.007        +0.001   -0.010   +0.001
#                          :    muoTimeAtIpInOut:  +0.018  -0.002      +0.038    +0.015    +0.001              +0.004      +0.010  +0.038     +0.009              -0.001              +0.065       +0.014       -0.005        +0.086          +0.063            -0.059        +0.000           +1.000        +0.042       +0.054    +0.000     -0.018        +0.101   -0.020   -0.007
#                          :       muoValPixHits:  -0.137  -0.043      +0.091    -0.028    -0.031              +0.011      +0.194  +0.516     +0.068              +0.013              +0.197       -0.024       -0.048        +0.255          +0.093            -0.101        +0.003           +0.042        +1.000       +0.637    -0.002     -0.143        +0.215   -0.003   -0.025
#                          :        muoNTrkVHits:  -0.159  +0.000      +0.077    -0.049    -0.033              +0.011      +0.252  +0.865     +0.098              +0.013              +0.188       -0.003       -0.033        +0.452          +0.068            -0.126        +0.001           +0.054        +0.637       +1.000    -0.002     -0.175        +0.177   +0.035   -0.019
#                          :           muoGNchi2:  -0.002  +0.001      -0.002    +0.002    +0.001              +0.003      -0.003  -0.002     -0.000              +0.001              +0.001       -0.000       +0.000        -0.001          -0.004            +0.005        +0.001           +0.000        -0.002       -0.002    +1.000     -0.003        -0.003   -0.002   +0.002
#                          :          muoVMuHits:  +0.276  +0.011      +0.236    -0.248    -0.044              -0.001      -0.025  -0.116     -0.035              +0.255              -0.330       -0.283       +0.030        -0.174          +0.758            -0.343        -0.007           -0.018        -0.143       -0.175    -0.003     +1.000        +0.368   +0.236   -0.011
#                          :       muoNumMatches:  +0.116  +0.021      +0.708    -0.096    -0.074              +0.003      +0.047  +0.121     +0.018              +0.079              +0.073       -0.153       -0.019        +0.272          +0.615            -0.425        +0.001           +0.101        +0.215       +0.177    -0.003     +0.368        +1.000   +0.085   -0.048
#                          :            muoQprod:  +0.057  +0.002      +0.068    -0.368    -0.041              -0.016      -0.005  +0.059     -0.003              +0.075              -0.077       -0.191       +0.003        -0.003          +0.257            -0.168        -0.010           -0.020        -0.003       +0.035    -0.002     +0.236        +0.085   +1.000   -0.008
#                          :            muoPFiso:  -0.041  +0.001      -0.031    +0.014    +0.009              +0.003      -0.013  -0.008     +0.002              +0.015              -0.015       +0.017       +0.006        -0.028          -0.036            +0.043        +0.001           -0.007        -0.025       -0.019    +0.002     -0.011        -0.048   -0.008   +1.000
#                          : --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# DataSetInfo              : Correlation matrix (Background):
#                          : --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                          :                        muoPt  muoEta muoSegmComp muoChi2LM muoChi2LP muoGlbTrackTailProb muoIValFrac  muoLWH muoTrkKink muoGlbKinkFinderLOG muoTimeAtIpInOutErr muoOuterChi2 muoInnerChi2 muoTrkRelChi2 muoVMuonHitComb muoGlbDeltaEtaPhi muoStaRelChi2 muoTimeAtIpInOut muoValPixHits muoNTrkVHits muoGNchi2 muoVMuHits muoNumMatches muoQprod muoPFiso
#                          :               muoPt:  +1.000  -0.004      -0.052    +0.145    +0.184              +0.126      -0.002  -0.098     -0.018              -0.108              -0.081       -0.042       +0.033        -0.034          +0.089            -0.171        +0.095           -0.022        -0.097       -0.148    +0.001     +0.193        +0.006   +0.014   +0.010
#                          :              muoEta:  -0.004  +1.000      +0.003    +0.005    +0.006              +0.004      +0.006  +0.004     -0.002              -0.000              -0.006       +0.002       -0.002        +0.006          +0.006            +0.003        +0.002           -0.002        -0.030       +0.011    +0.001     +0.004        +0.010   +0.001   -0.001
#                          :         muoSegmComp:  -0.052  +0.003      +1.000    -0.307    -0.280              -0.169      +0.009  +0.004     +0.107              +0.052              +0.091       -0.194       +0.111        +0.077          +0.386            -0.433        -0.084           -0.003        -0.097       +0.016    +0.001     +0.348        +0.800   +0.259   +0.042
#                          :           muoChi2LM:  +0.145  +0.005      -0.307    +1.000    +0.593              +0.629      +0.018  +0.034     -0.058              +0.066              -0.022       +0.114       -0.065        -0.003          -0.114            +0.333        +0.261           +0.050        +0.097       +0.029    -0.000     -0.123        -0.232   -0.452   -0.058
#                          :           muoChi2LP:  +0.184  +0.006      -0.280    +0.593    +1.000              +0.452      +0.004  +0.025     -0.042              +0.064              -0.018       +0.061       -0.049        +0.012          -0.050            +0.431        +0.177           +0.043        +0.082       +0.025    -0.001     -0.062        -0.196   -0.173   -0.051
#                          : muoGlbTrackTailProb:  +0.126  +0.004      -0.169    +0.629    +0.452              +1.000      +0.010  +0.034     +0.041              +0.187              +0.070       -0.043       +0.031        +0.078          +0.052            +0.179        +0.437           +0.042        +0.074       +0.041    -0.000     +0.033        -0.113   -0.236   -0.043
#                          :         muoIValFrac:  -0.002  +0.006      +0.009    +0.018    +0.004              +0.010      +1.000  +0.733     +0.082              -0.028              -0.005       +0.019       +0.021        +0.071          -0.008            +0.007        +0.005           -0.001        +0.294       +0.626    -0.003     -0.026        +0.026   -0.018   -0.063
#                          :              muoLWH:  -0.098  +0.004      +0.004    +0.034    +0.025              +0.034      +0.733  +1.000     +0.185              -0.009              +0.015       +0.028       +0.075        +0.239          -0.002            -0.001        +0.009           +0.014        +0.484       +0.902    -0.003     -0.082        +0.060   -0.027   -0.080
#                          :          muoTrkKink:  -0.018  -0.002      +0.107    -0.058    -0.042              +0.041      +0.082  +0.185     +1.000              +0.031              +0.054       -0.035       +0.826        +0.545          +0.093            -0.116        -0.013           +0.007        +0.124       +0.184    -0.000     +0.028        +0.141   +0.055   +0.004
#                          : muoGlbKinkFinderLOG:  -0.108  -0.000      +0.052    +0.066    +0.064              +0.187      -0.028  -0.009     +0.031              +1.000              +0.556       -0.421       +0.036        +0.081          +0.555            +0.079        +0.060           +0.033        +0.020       +0.008    -0.008     +0.465        +0.106   +0.040   +0.001
#                          : muoTimeAtIpInOutErr:  -0.081  -0.006      +0.091    -0.022    -0.018              +0.070      -0.005  +0.015     +0.054              +0.556              +1.000       -0.406       +0.036        +0.145          +0.290            -0.039        +0.078           -0.033        +0.053       +0.043    -0.005     +0.121        +0.125   +0.013   +0.008
#                          :        muoOuterChi2:  -0.042  +0.002      -0.194    +0.114    +0.061              -0.043      +0.019  +0.028     -0.035              -0.421              -0.406       +1.000       -0.045        -0.035          -0.438            +0.150        -0.022           +0.001        +0.036       +0.021    -0.001     -0.404        -0.211   -0.160   -0.035
#                          :        muoInnerChi2:  +0.033  -0.002      +0.111    -0.065    -0.049              +0.031      +0.021  +0.075     +0.826              +0.036              +0.036       -0.045       +1.000        +0.568          +0.096            -0.125        -0.015           +0.002        +0.028       +0.053    +0.005     +0.071        +0.125   +0.071   +0.019
#                          :       muoTrkRelChi2:  -0.034  +0.006      +0.077    -0.003    +0.012              +0.078      +0.071  +0.239     +0.545              +0.081              +0.145       -0.035       +0.568        +1.000          +0.144            -0.112        +0.012           +0.037        +0.148       +0.335    -0.004     -0.035        +0.207   -0.008   -0.027
#                          :     muoVMuonHitComb:  +0.089  +0.006      +0.386    -0.114    -0.050              +0.052      -0.008  -0.002     +0.093              +0.555              +0.290       -0.438       +0.096        +0.144          +1.000            -0.299        -0.029           +0.095        -0.016       +0.020    -0.005     +0.886        +0.524   +0.246   +0.026
#                          :   muoGlbDeltaEtaPhi:  -0.171  +0.003      -0.433    +0.333    +0.431              +0.179      +0.007  -0.001     -0.116              +0.079              -0.039       +0.150       -0.125        -0.112          -0.299            +1.000        +0.058           -0.005        +0.131       -0.013    -0.002     -0.292        -0.432   -0.249   -0.084
#                          :       muoStaRelChi2:  +0.095  +0.002      -0.084    +0.261    +0.177              +0.437      +0.005  +0.009     -0.013              +0.060              +0.078       -0.022       -0.015        +0.012          -0.029            +0.058        +1.000           +0.012        +0.024       +0.010    -0.000     -0.033        -0.064   -0.086   -0.017
#                          :    muoTimeAtIpInOut:  -0.022  -0.002      -0.003    +0.050    +0.043              +0.042      -0.001  +0.014     +0.007              +0.033              -0.033       +0.001       +0.002        +0.037          +0.095            -0.005        +0.012           +1.000        +0.026       +0.027    -0.000     +0.060        +0.040   -0.016   -0.010
#                          :       muoValPixHits:  -0.097  -0.030      -0.097    +0.097    +0.082              +0.074      +0.294  +0.484     +0.124              +0.020              +0.053       +0.036       +0.028        +0.148          -0.016            +0.131        +0.024           +0.026        +1.000       +0.516    -0.004     -0.130        +0.031   -0.128   -0.136
#                          :        muoNTrkVHits:  -0.148  +0.011      +0.016    +0.029    +0.025              +0.041      +0.626  +0.902     +0.184              +0.008              +0.043       +0.021       +0.053        +0.335          +0.020            -0.013        +0.010           +0.027        +0.516       +1.000    -0.004     -0.099        +0.101   -0.044   -0.081
#                          :           muoGNchi2:  +0.001  +0.001      +0.001    -0.000    -0.001              -0.000      -0.003  -0.003     -0.000              -0.008              -0.005       -0.001       +0.005        -0.004          -0.005            -0.002        -0.000           -0.000        -0.004       -0.004    +1.000     -0.004        -0.000   +0.001   +0.002
#                          :          muoVMuHits:  +0.193  +0.004      +0.348    -0.123    -0.062              +0.033      -0.026  -0.082     +0.028              +0.465              +0.121       -0.404       +0.071        -0.035          +0.886            -0.292        -0.033           +0.060        -0.130       -0.099    -0.004     +1.000        +0.422   +0.261   +0.052
#                          :       muoNumMatches:  +0.006  +0.010      +0.800    -0.232    -0.196              -0.113      +0.026  +0.060     +0.141              +0.106              +0.125       -0.211       +0.125        +0.207          +0.524            -0.432        -0.064           +0.040        +0.031       +0.101    -0.000     +0.422        +1.000   +0.219   +0.008
#                          :            muoQprod:  +0.014  +0.001      +0.259    -0.452    -0.173              -0.236      -0.018  -0.027     +0.055              +0.040              +0.013       -0.160       +0.071        -0.008          +0.246            -0.249        -0.086           -0.016        -0.128       -0.044    +0.001     +0.261        +0.219   +1.000   +0.054
#                          :            muoPFiso:  +0.010  -0.001      +0.042    -0.058    -0.051              -0.043      -0.063  -0.080     +0.004              +0.001              +0.008       -0.035       +0.019        -0.027          +0.026            -0.084        -0.017           -0.010        -0.136       -0.081    +0.002     +0.052        +0.008   +0.054   +1.000
#                          : --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# DataSetFactory           : [dataset] :  
#                          : 
# Factory                  : [dataset] : Create Transformation "I" with events from all classes.
#                          : 
#                          : Transformation, Variable selection : 
#                          : Input : variable 'muoPt' <---> Output : variable 'muoPt'
#                          : Input : variable 'muoEta' <---> Output : variable 'muoEta'
#                          : Input : variable 'muoSegmComp' <---> Output : variable 'muoSegmComp'
#                          : Input : variable 'muoChi2LM' <---> Output : variable 'muoChi2LM'
#                          : Input : variable 'muoChi2LP' <---> Output : variable 'muoChi2LP'
#                          : Input : variable 'muoGlbTrackTailProb' <---> Output : variable 'muoGlbTrackTailProb'
#                          : Input : variable 'muoIValFrac' <---> Output : variable 'muoIValFrac'
#                          : Input : variable 'muoLWH' <---> Output : variable 'muoLWH'
#                          : Input : variable 'muoTrkKink' <---> Output : variable 'muoTrkKink'
#                          : Input : variable 'muoGlbKinkFinderLOG' <---> Output : variable 'muoGlbKinkFinderLOG'
#                          : Input : variable 'muoTimeAtIpInOutErr' <---> Output : variable 'muoTimeAtIpInOutErr'
#                          : Input : variable 'muoOuterChi2' <---> Output : variable 'muoOuterChi2'
#                          : Input : variable 'muoInnerChi2' <---> Output : variable 'muoInnerChi2'
#                          : Input : variable 'muoTrkRelChi2' <---> Output : variable 'muoTrkRelChi2'
#                          : Input : variable 'muoVMuonHitComb' <---> Output : variable 'muoVMuonHitComb'
#                          : Input : variable 'muoGlbDeltaEtaPhi' <---> Output : variable 'muoGlbDeltaEtaPhi'
#                          : Input : variable 'muoStaRelChi2' <---> Output : variable 'muoStaRelChi2'
#                          : Input : variable 'muoTimeAtIpInOut' <---> Output : variable 'muoTimeAtIpInOut'
#                          : Input : variable 'muoValPixHits' <---> Output : variable 'muoValPixHits'
#                          : Input : variable 'muoNTrkVHits' <---> Output : variable 'muoNTrkVHits'
#                          : Input : variable 'muoGNchi2' <---> Output : variable 'muoGNchi2'
#                          : Input : variable 'muoVMuHits' <---> Output : variable 'muoVMuHits'
#                          : Input : variable 'muoNumMatches' <---> Output : variable 'muoNumMatches'
#                          : Input : variable 'muoQprod' <---> Output : variable 'muoQprod'
#                          : Input : variable 'muoPFiso' <---> Output : variable 'muoPFiso'
# TFHandler_Factory        :            Variable                   Mean                   RMS           [        Min                   Max ]
#                          : ------------------------------------------------------------------------------------------------------------------
#                          :               muoPt:               4.9682               3.5975   [               2.0000               255.54 ]
#                          :              muoEta:            0.0029664               1.5401   [              -2.4000               2.4000 ]
#                          :         muoSegmComp:              0.75301              0.26836   [               0.0000               1.0000 ]
#                          :           muoChi2LM:               61.348               213.76   [              -1.0000               4998.3 ]
#                          :           muoChi2LP:               9.4033               41.757   [              -1.0000               1990.8 ]
#                          : muoGlbTrackTailProb:               21.307               129.94   [               0.0000               4991.3 ]
#                          :         muoIValFrac:              0.95920             0.082669   [              0.36842               1.0000 ]
#                          :              muoLWH:               13.023               2.2808   [               4.0000               18.000 ]
#                          :          muoTrkKink:               14.267               23.875   [             0.010675               867.77 ]
#                          : muoGlbKinkFinderLOG:               10.747               2.6933   [              0.69315               48.348 ]
#                          : muoTimeAtIpInOutErr:               1.1943              0.59507   [               0.0000               4.0000 ]
#                          :        muoOuterChi2:               7.5963               34.619   [           0.00073940               999.71 ]
#                          :        muoInnerChi2:               1.3100              0.85713   [             0.028275               9.9967 ]
#                          :       muoTrkRelChi2:              0.45681              0.29983   [               0.0000               3.0000 ]
#                          :     muoVMuonHitComb:               17.616               7.8059   [               0.0000               34.000 ]
#                          :   muoGlbDeltaEtaPhi:             0.051926             0.063039   [              -1.0000               2.2402 ]
#                          :       muoStaRelChi2:               1.2919               13.442   [               0.0000               8010.6 ]
#                          :    muoTimeAtIpInOut:             -0.40377               17.171   [              -2494.5               389.24 ]
#                          :       muoValPixHits:               4.1000               1.6420   [               0.0000               11.000 ]
#                          :        muoNTrkVHits:               19.010               4.1418   [               7.0000               35.000 ]
#                          :           muoGNchi2:               46.965               21923.   [             0.046422           1.7129e+07 ]
#                          :          muoVMuHits:               21.710               10.947   [               0.0000               52.000 ]
#                          :       muoNumMatches:               2.9053               1.0540   [               0.0000               5.0000 ]
#                          :            muoQprod:              0.84886              0.52861   [              -1.0000               1.0000 ]
#                          :            muoPFiso:               2.0804               3.1493   [               0.0000               1225.5 ]
#                          : ------------------------------------------------------------------------------------------------------------------
#                          : 
#                          : <PlotVariables> Will not produce scatter plots ==> 
#                          : |  The number of 25 input variables and 0 target values would require 300 two-dimensional
#                          : |  histograms, which would occupy the computer's memory. Note that this
#                          : |  suppression does not have any consequences for your analysis, other
#                          : |  than not disposing of these scatter plots. You can modify the maximum
#                          : |  number of input variables allowed to generate scatter plots in your
#                          : |  script via the command line:
#                          : |  "(TMVA::gConfig().GetVariablePlotting()).fMaxNumOfAllowedVariablesForScatterPlots = <some int>;"
#                          : 
#                          : Some more output
#                          : Ranking input variables (method unspecific)...
# IdTransformation         : Ranking result (top variable is best ranked)
#                          : --------------------------------------------
#                          : Rank : Variable            : Separation
#                          : --------------------------------------------
#                          :    1 : muoGlbDeltaEtaPhi   : 1.684e-01
#                          :    2 : muoIValFrac         : 1.542e-01
#                          :    3 : muoPFiso            : 1.488e-01
#                          :    4 : muoChi2LP           : 1.429e-01
#                          :    5 : muoChi2LM           : 1.108e-01
#                          :    6 : muoSegmComp         : 1.093e-01
#                          :    7 : muoGlbKinkFinderLOG : 1.024e-01
#                          :    8 : muoTrkKink          : 9.422e-02
#                          :    9 : muoGlbTrackTailProb : 8.983e-02
#                          :   10 : muoTimeAtIpInOut    : 8.979e-02
#                          :   11 : muoValPixHits       : 8.398e-02
#                          :   12 : muoNumMatches       : 7.610e-02
#                          :   13 : muoPt               : 7.373e-02
#                          :   14 : muoTimeAtIpInOutErr : 7.176e-02
#                          :   15 : muoInnerChi2        : 7.152e-02
#                          :   16 : muoVMuHits          : 7.018e-02
#                          :   17 : muoLWH              : 5.810e-02
#                          :   18 : muoTrkRelChi2       : 5.341e-02
#                          :   19 : muoVMuonHitComb     : 4.744e-02
#                          :   20 : muoEta              : 3.946e-02
#                          :   21 : muoNTrkVHits        : 3.841e-02
#                          :   22 : muoQprod            : 3.521e-02
#                          :   23 : muoStaRelChi2       : 2.872e-02
#                          :   24 : muoOuterChi2        : 2.698e-02
#                          :   25 : muoGNchi2           : 2.202e-05
#                          : --------------------------------------------
# Factory                  : Train method: DNNMuonID for Classification
#                          : 
#                          : Preparing the Gaussian transformation...
# TFHandler_DNNMuonID      :            Variable                   Mean                   RMS           [        Min                   Max ]
#                          : ------------------------------------------------------------------------------------------------------------------
#                          :               muoPt:             -0.29574              0.22502   [              -1.0000               1.0000 ]
#                          :              muoEta:             -0.27212              0.22138   [              -1.0000               1.0000 ]
#                          :         muoSegmComp:            -0.031055              0.54699   [              -1.0000               1.0000 ]
#                          :           muoChi2LM:             -0.27598              0.22119   [              -1.0000               1.0000 ]
#                          :           muoChi2LP:             -0.26681              0.21956   [              -1.0000               1.0000 ]
#                          : muoGlbTrackTailProb:             -0.29931              0.23731   [              -1.0000               1.0000 ]
#                          :         muoIValFrac:              0.61836              0.53309   [              -1.0000               1.0000 ]
#                          :              muoLWH:            -0.086059              0.19017   [              -1.0000               1.0000 ]
#                          :          muoTrkKink:             -0.27427              0.22144   [              -1.0000               1.0000 ]
#                          : muoGlbKinkFinderLOG:             -0.29948              0.22665   [              -1.0000               1.0000 ]
#                          : muoTimeAtIpInOutErr:             -0.35879              0.29271   [              -1.0000               1.0000 ]
#                          :        muoOuterChi2:             -0.28396              0.22246   [              -1.0000               1.0000 ]
#                          :        muoInnerChi2:             -0.27001              0.22087   [              -1.0000               1.0000 ]
#                          :       muoTrkRelChi2:             -0.27093              0.22106   [              -1.0000               1.0000 ]
#                          :     muoVMuonHitComb:             -0.38466              0.28166   [              -1.0000               1.0000 ]
#                          :   muoGlbDeltaEtaPhi:             -0.26995              0.22055   [              -1.0000               1.0000 ]
#                          :       muoStaRelChi2:             -0.33570              0.27190   [              -1.0000               1.0000 ]
#                          :    muoTimeAtIpInOut:             -0.26989              0.21181   [              -1.0000               1.0000 ]
#                          :       muoValPixHits:             -0.42681              0.22676   [              -1.0000               1.0000 ]
#                          :        muoNTrkVHits:             -0.25454              0.21719   [              -1.0000               1.0000 ]
#                          :           muoGNchi2:             -0.99341             0.036721   [              -1.0000               1.0000 ]
#                          :          muoVMuHits:             -0.40636              0.27937   [              -1.0000               1.0000 ]
#                          :       muoNumMatches:             -0.16916              0.13157   [              -1.0000               1.0000 ]
#                          :            muoQprod:              0.84886              0.52861   [              -1.0000               1.0000 ]
#                          :            muoPFiso:             -0.37644              0.30262   [              -1.0000               1.0000 ]
#                          : ------------------------------------------------------------------------------------------------------------------
# TFHandler_DNNMuonID      :            Variable                   Mean                   RMS           [        Min                   Max ]
#                          : ------------------------------------------------------------------------------------------------------------------
#                          :               muoPt:             -0.29574              0.22502   [              -1.0000               1.0000 ]
#                          :              muoEta:             -0.27212              0.22138   [              -1.0000               1.0000 ]
#                          :         muoSegmComp:            -0.031055              0.54699   [              -1.0000               1.0000 ]
#                          :           muoChi2LM:             -0.27598              0.22119   [              -1.0000               1.0000 ]
#                          :           muoChi2LP:             -0.26681              0.21956   [              -1.0000               1.0000 ]
#                          : muoGlbTrackTailProb:             -0.29931              0.23731   [              -1.0000               1.0000 ]
#                          :         muoIValFrac:              0.61836              0.53309   [              -1.0000               1.0000 ]
#                          :              muoLWH:            -0.086059              0.19017   [              -1.0000               1.0000 ]
#                          :          muoTrkKink:             -0.27427              0.22144   [              -1.0000               1.0000 ]
#                          : muoGlbKinkFinderLOG:             -0.29948              0.22665   [              -1.0000               1.0000 ]
#                          : muoTimeAtIpInOutErr:             -0.35879              0.29271   [              -1.0000               1.0000 ]
#                          :        muoOuterChi2:             -0.28396              0.22246   [              -1.0000               1.0000 ]
#                          :        muoInnerChi2:             -0.27001              0.22087   [              -1.0000               1.0000 ]
#                          :       muoTrkRelChi2:             -0.27093              0.22106   [              -1.0000               1.0000 ]
#                          :     muoVMuonHitComb:             -0.38466              0.28166   [              -1.0000               1.0000 ]
#                          :   muoGlbDeltaEtaPhi:             -0.26995              0.22055   [              -1.0000               1.0000 ]
#                          :       muoStaRelChi2:             -0.33570              0.27190   [              -1.0000               1.0000 ]
#                          :    muoTimeAtIpInOut:             -0.26989              0.21181   [              -1.0000               1.0000 ]
#                          :       muoValPixHits:             -0.42681              0.22676   [              -1.0000               1.0000 ]
#                          :        muoNTrkVHits:             -0.25454              0.21719   [              -1.0000               1.0000 ]
#                          :           muoGNchi2:             -0.99341             0.036721   [              -1.0000               1.0000 ]
#                          :          muoVMuHits:             -0.40636              0.27937   [              -1.0000               1.0000 ]
#                          :       muoNumMatches:             -0.16916              0.13157   [              -1.0000               1.0000 ]
#                          :            muoQprod:              0.84886              0.52861   [              -1.0000               1.0000 ]
#                          :            muoPFiso:             -0.37644              0.30262   [              -1.0000               1.0000 ]
#                          : ------------------------------------------------------------------------------------------------------------------
#                          : Split TMVA training data in 3556836 training events and 1751874 validation events
#                          : Option SaveBestOnly: Only model weights with smallest validation loss will be stored
#                          : Option TriesEarlyStopping: Training will stop after 10 number of epochs with no improvement of validation loss
#                          : Option TensorBoard: Log files for training monitoring are stored in: './logs'
# WARNING:tensorflow:From /home/alberto/.local/lib/python2.7/site-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
# Instructions for updating:
# Use tf.cast instead.
# Train on 3556836 samples, validate on 1751874 samples
# Epoch 1/40
# 3556836/3556836 [==============================] - 19s 5us/step - loss: 0.6498 - acc: 0.8803 - val_loss: 0.6204 - val_acc: 0.9006

# Epoch 00001: val_loss improved from inf to 0.62036, saving model to dataset/weights/TrainedModel_DNNMuonID.h5
# Epoch 2/40
# 3556836/3556836 [==============================] - 17s 5us/step - loss: 0.6218 - acc: 0.8898 - val_loss: 0.6115 - val_acc: 0.8801

# Epoch 00002: val_loss improved from 0.62036 to 0.61155, saving model to dataset/weights/TrainedModel_DNNMuonID.h5
# Epoch 3/40
# 3556836/3556836 [==============================] - 18s 5us/step - loss: 0.6153 - acc: 0.8918 - val_loss: 0.6074 - val_acc: 0.9058

# Epoch 00003: val_loss improved from 0.61155 to 0.60742, saving model to dataset/weights/TrainedModel_DNNMuonID.h5
# Epoch 4/40
# 3556836/3556836 [==============================] - 18s 5us/step - loss: 0.6117 - acc: 0.8931 - val_loss: 0.6064 - val_acc: 0.9002

# Epoch 00004: val_loss improved from 0.60742 to 0.60637, saving model to dataset/weights/TrainedModel_DNNMuonID.h5
# Epoch 5/40
# 3556836/3556836 [==============================] - 18s 5us/step - loss: 0.6090 - acc: 0.8940 - val_loss: 0.6040 - val_acc: 0.8905

# Epoch 00005: val_loss improved from 0.60637 to 0.60400, saving model to dataset/weights/TrainedModel_DNNMuonID.h5
# Epoch 6/40
# 3556836/3556836 [==============================] - 18s 5us/step - loss: 0.6077 - acc: 0.8949 - val_loss: 0.6032 - val_acc: 0.8950

# Epoch 00006: val_loss improved from 0.60400 to 0.60319, saving model to dataset/weights/TrainedModel_DNNMuonID.h5
# Epoch 7/40
# 3556836/3556836 [==============================] - 18s 5us/step - loss: 0.6061 - acc: 0.8954 - val_loss: 0.6008 - val_acc: 0.8948

# Epoch 00007: val_loss improved from 0.60319 to 0.60085, saving model to dataset/weights/TrainedModel_DNNMuonID.h5
# Epoch 8/40
# 3556836/3556836 [==============================] - 18s 5us/step - loss: 0.6049 - acc: 0.8954 - val_loss: 0.5992 - val_acc: 0.9066

# Epoch 00008: val_loss improved from 0.60085 to 0.59916, saving model to dataset/weights/TrainedModel_DNNMuonID.h5
# Epoch 9/40
# 3556836/3556836 [==============================] - 18s 5us/step - loss: 0.6038 - acc: 0.8956 - val_loss: 0.6029 - val_acc: 0.9119

# Epoch 00009: val_loss did not improve
# Epoch 10/40
# 3556836/3556836 [==============================] - 18s 5us/step - loss: 0.6027 - acc: 0.8965 - val_loss: 0.5986 - val_acc: 0.9053

# Epoch 00010: val_loss improved from 0.59916 to 0.59865, saving model to dataset/weights/TrainedModel_DNNMuonID.h5
# Epoch 11/40
# 3556836/3556836 [==============================] - 18s 5us/step - loss: 0.6024 - acc: 0.8965 - val_loss: 0.5986 - val_acc: 0.9009

# Epoch 00011: val_loss improved from 0.59865 to 0.59858, saving model to dataset/weights/TrainedModel_DNNMuonID.h5
# Epoch 12/40
# 3556836/3556836 [==============================] - 18s 5us/step - loss: 0.6017 - acc: 0.8964 - val_loss: 0.5983 - val_acc: 0.9048

# Epoch 00012: val_loss improved from 0.59858 to 0.59834, saving model to dataset/weights/TrainedModel_DNNMuonID.h5
# Epoch 13/40
# 3556836/3556836 [==============================] - 18s 5us/step - loss: 0.6010 - acc: 0.8968 - val_loss: 0.5983 - val_acc: 0.8880

# Epoch 00013: val_loss improved from 0.59834 to 0.59826, saving model to dataset/weights/TrainedModel_DNNMuonID.h5
# Epoch 14/40
# 3556836/3556836 [==============================] - 18s 5us/step - loss: 0.6004 - acc: 0.8972 - val_loss: 0.5948 - val_acc: 0.8990

# Epoch 00014: val_loss improved from 0.59826 to 0.59480, saving model to dataset/weights/TrainedModel_DNNMuonID.h5
# Epoch 15/40
# 3556836/3556836 [==============================] - 18s 5us/step - loss: 0.5999 - acc: 0.8974 - val_loss: 0.5958 - val_acc: 0.8946

# Epoch 00015: val_loss did not improve
# Epoch 16/40
# 3556836/3556836 [==============================] - 18s 5us/step - loss: 0.5997 - acc: 0.8973 - val_loss: 0.5953 - val_acc: 0.9059

# Epoch 00016: val_loss did not improve
# Epoch 17/40
# 3556836/3556836 [==============================] - 18s 5us/step - loss: 0.5988 - acc: 0.8976 - val_loss: 0.5960 - val_acc: 0.9039

# Epoch 00017: val_loss did not improve
# Epoch 18/40
# 3556836/3556836 [==============================] - 18s 5us/step - loss: 0.5987 - acc: 0.8977 - val_loss: 0.5957 - val_acc: 0.9054

# Epoch 00018: val_loss did not improve
# Epoch 19/40
# 3556836/3556836 [==============================] - 18s 5us/step - loss: 0.5981 - acc: 0.8979 - val_loss: 0.5972 - val_acc: 0.8873

# Epoch 00019: val_loss did not improve
# Epoch 20/40
# 3556836/3556836 [==============================] - 18s 5us/step - loss: 0.5976 - acc: 0.8981 - val_loss: 0.5938 - val_acc: 0.8966

# Epoch 00020: val_loss improved from 0.59480 to 0.59380, saving model to dataset/weights/TrainedModel_DNNMuonID.h5
# Epoch 21/40
# 3556836/3556836 [==============================] - 18s 5us/step - loss: 0.5975 - acc: 0.8979 - val_loss: 0.5966 - val_acc: 0.8877

# Epoch 00021: val_loss did not improve
# Epoch 22/40
# 3556836/3556836 [==============================] - 18s 5us/step - loss: 0.5969 - acc: 0.8982 - val_loss: 0.5944 - val_acc: 0.9048

# Epoch 00022: val_loss did not improve
# Epoch 23/40
# 3556836/3556836 [==============================] - 18s 5us/step - loss: 0.5966 - acc: 0.8984 - val_loss: 0.5934 - val_acc: 0.9069

# Epoch 00023: val_loss improved from 0.59380 to 0.59337, saving model to dataset/weights/TrainedModel_DNNMuonID.h5
# Epoch 24/40
# 3556836/3556836 [==============================] - 18s 5us/step - loss: 0.5965 - acc: 0.8982 - val_loss: 0.5939 - val_acc: 0.9059

# Epoch 00024: val_loss did not improve
# Epoch 25/40
# 3556836/3556836 [==============================] - 18s 5us/step - loss: 0.5961 - acc: 0.8986 - val_loss: 0.5948 - val_acc: 0.9075

# Epoch 00025: val_loss did not improve
# Epoch 26/40
# 3556836/3556836 [==============================] - 19s 5us/step - loss: 0.5960 - acc: 0.8988 - val_loss: 0.5934 - val_acc: 0.9050

# Epoch 00026: val_loss did not improve
# Epoch 27/40
# 3556836/3556836 [==============================] - 19s 5us/step - loss: 0.5954 - acc: 0.8985 - val_loss: 0.5943 - val_acc: 0.9053

# Epoch 00027: val_loss did not improve
# Epoch 28/40
# 3556836/3556836 [==============================] - 18s 5us/step - loss: 0.5954 - acc: 0.8988 - val_loss: 0.5945 - val_acc: 0.9060

# Epoch 00028: val_loss did not improve
# Epoch 29/40
# 3556836/3556836 [==============================] - 18s 5us/step - loss: 0.5948 - acc: 0.8988 - val_loss: 0.5936 - val_acc: 0.9038

# Epoch 00029: val_loss did not improve
# Epoch 30/40
# 3556836/3556836 [==============================] - 18s 5us/step - loss: 0.5946 - acc: 0.8990 - val_loss: 0.5940 - val_acc: 0.8934

# Epoch 00030: val_loss did not improve
# Epoch 31/40
# 3556836/3556836 [==============================] - 18s 5us/step - loss: 0.5944 - acc: 0.8991 - val_loss: 0.5927 - val_acc: 0.9038

# Epoch 00031: val_loss improved from 0.59337 to 0.59269, saving model to dataset/weights/TrainedModel_DNNMuonID.h5
# Epoch 32/40
# 3556836/3556836 [==============================] - 18s 5us/step - loss: 0.5942 - acc: 0.8988 - val_loss: 0.5943 - val_acc: 0.8975

# Epoch 00032: val_loss did not improve
# Epoch 33/40
# 3556836/3556836 [==============================] - 18s 5us/step - loss: 0.5944 - acc: 0.8991 - val_loss: 0.5981 - val_acc: 0.8830

# Epoch 00033: val_loss did not improve
# Epoch 34/40
# 3556836/3556836 [==============================] - 18s 5us/step - loss: 0.5941 - acc: 0.8988 - val_loss: 0.5938 - val_acc: 0.9094

# Epoch 00034: val_loss did not improve
# Epoch 35/40
# 3556836/3556836 [==============================] - 18s 5us/step - loss: 0.5939 - acc: 0.8990 - val_loss: 0.5933 - val_acc: 0.9013

# Epoch 00035: val_loss did not improve
# Epoch 36/40
# 3556836/3556836 [==============================] - 18s 5us/step - loss: 0.5933 - acc: 0.8988 - val_loss: 0.5958 - val_acc: 0.9095

# Epoch 00036: val_loss did not improve
# Epoch 37/40
# 3556836/3556836 [==============================] - 19s 5us/step - loss: 0.5930 - acc: 0.8988 - val_loss: 0.5930 - val_acc: 0.9097

# Epoch 00037: val_loss did not improve
# Epoch 38/40
# 3556836/3556836 [==============================] - 18s 5us/step - loss: 0.5934 - acc: 0.8990 - val_loss: 0.5945 - val_acc: 0.8934

# Epoch 00038: val_loss did not improve
# Epoch 39/40
# 3556836/3556836 [==============================] - 18s 5us/step - loss: 0.5932 - acc: 0.8994 - val_loss: 0.5921 - val_acc: 0.8987

# Epoch 00039: val_loss improved from 0.59269 to 0.59208, saving model to dataset/weights/TrainedModel_DNNMuonID.h5
# Epoch 40/40
# 3556836/3556836 [==============================] - 18s 5us/step - loss: 0.5930 - acc: 0.8993 - val_loss: 0.5944 - val_acc: 0.9065

# Epoch 00040: val_loss did not improve
#                          : Elapsed time for training with 5308710 events: 839 sec         
# DNNMuonID                : [dataset] : Evaluation of DNNMuonID on training sample (5308710 events)
#                          : Elapsed time for evaluation of 5308710 events: 101 sec       
#                          : Creating xml weight file: dataset/weights/TMVAClassification_DNNMuonID.weights.xml
#                          : Creating standalone class: dataset/weights/TMVAClassification_DNNMuonID.class.C
# Factory                  : Training finished
#                          : 
#                          : Ranking input variables (method specific)...
#                          : No variable ranking supplied by classifier: DNNMuonID
# Factory                  : === Destroy and recreate all methods via weight files for testing ===
#                          : 
#                          : Reading weight file: dataset/weights/TMVAClassification_DNNMuonID.weights.xml
# <WARNING>                : Value for option tensorboard was previously set to ./logs
# Factory                  : Test all methods
# Factory                  : Test method: DNNMuonID for Classification performance
#                          : 
#                          : Load model from file: dataset/weights/TrainedModel_DNNMuonID.h5
# DNNMuonID                : [dataset] : Evaluation of DNNMuonID on testing sample (1769520 events)
#                          : Elapsed time for evaluation of 1769520 events: 34.2 sec       
# Factory                  : Evaluate all methods
# Factory                  : Evaluate classifier: DNNMuonID
#                          : 
# TFHandler_DNNMuonID      :            Variable                   Mean                   RMS           [        Min                   Max ]
#                          : ------------------------------------------------------------------------------------------------------------------
#                          :               muoPt:             -0.28434              0.22557   [             -0.99985               1.0000 ]
#                          :              muoEta:             -0.27191              0.21824   [             -0.99982               1.0000 ]
#                          :         muoSegmComp:           -0.0050317              0.55201   [              -1.0000               1.0000 ]
#                          :           muoChi2LM:             -0.29232              0.21725   [              -1.0000               1.0000 ]
#                          :           muoChi2LP:             -0.28161              0.21366   [              -1.0000               1.0000 ]
#                          : muoGlbTrackTailProb:             -0.31829              0.23288   [             -0.99958               1.0000 ]
#                          :         muoIValFrac:              0.65241              0.51591   [              -1.1640               1.0000 ]
#                          :              muoLWH:            -0.080719              0.18639   [              -1.0000               1.0000 ]
#                          :          muoTrkKink:             -0.28307              0.21441   [              -1.0027               1.0000 ]
#                          : muoGlbKinkFinderLOG:             -0.30699              0.21999   [              -1.0000               1.0000 ]
#                          : muoTimeAtIpInOutErr:             -0.36108              0.28591   [              -1.0000               1.0000 ]
#                          :        muoOuterChi2:             -0.28946              0.21874   [             -0.99961               1.0000 ]
#                          :        muoInnerChi2:             -0.27918              0.21618   [              -1.0178               1.0000 ]
#                          :       muoTrkRelChi2:             -0.27904              0.21802   [              -1.0000               1.0000 ]
#                          :     muoVMuonHitComb:             -0.37497              0.27674   [              -1.0000               1.0000 ]
#                          :   muoGlbDeltaEtaPhi:             -0.28688              0.21733   [              -1.0000               1.0000 ]
#                          :       muoStaRelChi2:             -0.33548              0.26027   [             -0.99905               1.0000 ]
#                          :    muoTimeAtIpInOut:             -0.27040              0.20482   [              -1.5473               1.0000 ]
#                          :       muoValPixHits:             -0.42251              0.21827   [              -1.0000               1.0000 ]
#                          :        muoNTrkVHits:             -0.25152              0.21315   [              -1.0000               1.0000 ]
#                          :           muoGNchi2:             -0.99407             0.034890   [             -0.99997               1.0000 ]
#                          :          muoVMuHits:             -0.39340              0.27650   [              -1.0000               1.0000 ]
#                          :       muoNumMatches:             -0.16279              0.12915   [              -1.0000               1.0000 ]
#                          :            muoQprod:              0.86767              0.49713   [              -1.0000               1.0000 ]
#                          :            muoPFiso:             -0.39561              0.30125   [             -0.99994              0.44888 ]
#                          : ------------------------------------------------------------------------------------------------------------------
# DNNMuonID                : [dataset] : Loop over test events and fill histograms with classifier response...
#                          : 
# TFHandler_DNNMuonID      :            Variable                   Mean                   RMS           [        Min                   Max ]
#                          : ------------------------------------------------------------------------------------------------------------------
#                          :               muoPt:             -0.28434              0.22557   [             -0.99985               1.0000 ]
#                          :              muoEta:             -0.27191              0.21824   [             -0.99982               1.0000 ]
#                          :         muoSegmComp:           -0.0050317              0.55201   [              -1.0000               1.0000 ]
#                          :           muoChi2LM:             -0.29232              0.21725   [              -1.0000               1.0000 ]
#                          :           muoChi2LP:             -0.28161              0.21366   [              -1.0000               1.0000 ]
#                          : muoGlbTrackTailProb:             -0.31829              0.23288   [             -0.99958               1.0000 ]
#                          :         muoIValFrac:              0.65241              0.51591   [              -1.1640               1.0000 ]
#                          :              muoLWH:            -0.080719              0.18639   [              -1.0000               1.0000 ]
#                          :          muoTrkKink:             -0.28307              0.21441   [              -1.0027               1.0000 ]
#                          : muoGlbKinkFinderLOG:             -0.30699              0.21999   [              -1.0000               1.0000 ]
#                          : muoTimeAtIpInOutErr:             -0.36108              0.28591   [              -1.0000               1.0000 ]
#                          :        muoOuterChi2:             -0.28946              0.21874   [             -0.99961               1.0000 ]
#                          :        muoInnerChi2:             -0.27918              0.21618   [              -1.0178               1.0000 ]
#                          :       muoTrkRelChi2:             -0.27904              0.21802   [              -1.0000               1.0000 ]
#                          :     muoVMuonHitComb:             -0.37497              0.27674   [              -1.0000               1.0000 ]
#                          :   muoGlbDeltaEtaPhi:             -0.28688              0.21733   [              -1.0000               1.0000 ]
#                          :       muoStaRelChi2:             -0.33548              0.26027   [             -0.99905               1.0000 ]
#                          :    muoTimeAtIpInOut:             -0.27040              0.20482   [              -1.5473               1.0000 ]
#                          :       muoValPixHits:             -0.42251              0.21827   [              -1.0000               1.0000 ]
#                          :        muoNTrkVHits:             -0.25152              0.21315   [              -1.0000               1.0000 ]
#                          :           muoGNchi2:             -0.99407             0.034890   [             -0.99997               1.0000 ]
#                          :          muoVMuHits:             -0.39340              0.27650   [              -1.0000               1.0000 ]
#                          :       muoNumMatches:             -0.16279              0.12915   [              -1.0000               1.0000 ]
#                          :            muoQprod:              0.86767              0.49713   [              -1.0000               1.0000 ]
#                          :            muoPFiso:             -0.39561              0.30125   [             -0.99994              0.44888 ]
#                          : ------------------------------------------------------------------------------------------------------------------
#                          : 
#                          : <PlotVariables> Will not produce scatter plots ==> 
#                          : |  The number of 25 input variables and 0 target values would require 300 two-dimensional
#                          : |  histograms, which would occupy the computer's memory. Note that this
#                          : |  suppression does not have any consequences for your analysis, other
#                          : |  than not disposing of these scatter plots. You can modify the maximum
#                          : |  number of input variables allowed to generate scatter plots in your
#                          : |  script via the command line:
#                          : |  "(TMVA::gConfig().GetVariablePlotting()).fMaxNumOfAllowedVariablesForScatterPlots = <some int>;"
#                          : 
#                          : Some more output
#                          : 
#                          : Evaluation results ranked by best signal efficiency and purity (area)
#                          : -------------------------------------------------------------------------------------------------------------------
#                          : DataSet       MVA                       
#                          : Name:         Method:          ROC-integ
#                          : dataset       DNNMuonID      : 0.934
#                          : -------------------------------------------------------------------------------------------------------------------
#                          : 
#                          : Testing efficiency compared to training efficiency (overtraining check)
#                          : -------------------------------------------------------------------------------------------------------------------
#                          : DataSet              MVA              Signal efficiency: from test sample (from training sample) 
#                          : Name:                Method:          @B=0.01             @B=0.10            @B=0.30   
#                          : -------------------------------------------------------------------------------------------------------------------
#                          : dataset              DNNMuonID      : 0.258 (0.277)       0.771 (0.779)      0.975 (0.976)
#                          : -------------------------------------------------------------------------------------------------------------------
#                          : 
# Dataset:dataset          : Created tree 'TestTree' with 1769520 events
#                          : 
# Dataset:dataset          : Created tree 'TrainTree' with 5308710 events
