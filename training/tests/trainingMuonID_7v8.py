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
    model.add(Dropout(dropValue))

    for i in range(1, nLayers):
        model.add(Dense(layerSize, activation='relu', kernel_initializer='normal'))
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

nLayers = 3
layerSize = 200
dropValue = 0.2

dropValue = float(dropValue)
layerSize = int(layerSize)
nLayers = int(nLayers)

isoflag = 'wIso'

if isoflag != 'wIso' and isoflag != 'woIso':
    print 'Invalid argument isoflag'
    sys.exit()


outputName = 'TMVAMuonID_7v8.root'
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

mycuts = mycutgen + '1'
mycutb = mycutgen + '1'

dataloader.AddSignalTree(treeBs17)
dataloader.AddSignalTree(treeBsD017)
dataloader.AddSignalTree(treeBu17)
dataloader.AddSignalTree(treeBd17)

dataloader.AddBackgroundTree(treeBs18)
dataloader.AddBackgroundTree(treeBsD018)
dataloader.AddBackgroundTree(treeBu18)
dataloader.AddBackgroundTree(treeBd18)


nBkg = '0'
nSgn = '0'
nBkgTest = '0'
nSgnTest = '0'


dataloaderOpt = 'nTrain_Signal=' + nSgn + ':nTrain_Background=' + nBkg + ':nTest_Signal=' + nSgnTest + ':nTest_Background=' + nBkgTest
dataloaderOpt += ':SplitMode=Random:NormMode=EqualNumEvents:!V'

dataloader.PrepareTrainingAndTestTree(TCut(mycuts), TCut(mycutb), dataloaderOpt)

# Define model
modelName = 'model17'
if isoflag == 'woIso':
    modelName += isoflag

modelName += '.h5'

getKerasModel(nVars, modelName, nLayers, layerSize, dropValue)

# Book methods
dnnOptions = '!H:!V:FilenameModel=' + modelName + ':NumEpochs=20:TriesEarlyStopping=10:BatchSize=512:ValidationSize=30%:Tensorboard=./logs'

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

dnnName = 'DNNMuonID_7p8'
if isoflag == 'woIso':
    dnnName += isoflag

factory.BookMethod(dataloader, TMVA.Types.kPyKeras, dnnName, dnnOptions + preprocessingOptions)


# Run training, test and evaluation
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()

#TMVA.TMVAGui("TMVA_keras.root")
