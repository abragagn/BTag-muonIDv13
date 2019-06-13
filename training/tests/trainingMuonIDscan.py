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

name = 'FullScanDP2_fine'

outputName = 'TMVAMuonID' + name + '.root'
output = TFile.Open(outputName, 'RECREATE')

# Factory
factory = TMVA.Factory('TMVAClassification', output,
                       '!V:!Silent:Color:DrawProgressBar:AnalysisType=Classification')

# Load data
#2017
dataBs17 = TFile.Open('../bankBsJpsiPhi17.root')
dataBsD017 = TFile.Open('../bankBsJpsiPhiDG017.root')
dataBu17 = TFile.Open('../bankBuJpsiK17.root')
dataBd17 = TFile.Open('../bankBdJpsiKx17.root')

treeBs17 = dataBs17.Get('PDsecondTree')
treeBsD017 = dataBsD017.Get('PDsecondTree')
treeBu17 = dataBu17.Get('PDsecondTree')
treeBd17 = dataBd17.Get('PDsecondTree')

#2018
dataBs18 = TFile.Open('../bankBsJpsiPhi18.root')
dataBsD018 = TFile.Open('../bankBsJpsiPhiDG018.root')
dataBu18 = TFile.Open('../bankBuJpsiK18.root')
dataBd18 = TFile.Open('../bankBdJpsiKx18.root')

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
    ,('muoPFiso', 'F')
    ]

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

nBkg = '0'
nSgn = '0'
nBkgTest = '0'
nSgnTest = '0'

dataloaderOpt = 'nTrain_Signal=' + nSgn + ':nTrain_Background=' + nBkg + ':nTest_Signal=' + nSgnTest + ':nTest_Background=' + nBkgTest
dataloaderOpt += ':SplitMode=Random:NormMode=EqualNumEvents:!V'

dataloader.PrepareTrainingAndTestTree(TCut(mycuts), TCut(mycutb), dataloaderOpt)

# Define model
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

ntu_nLayers = (2, 3)
ntu_layerSize = (100, 200)
ntu_dropValue = (0.0, 0.1, 0.2)

for nLayers in ntu_nLayers:
    for layerSize in ntu_layerSize:
        dropValue = ntu_dropValue[2]
        suffix = '_' + str(nLayers) + '_' + str(layerSize) + '_' + str(int(dropValue*10))
        modelName = 'model' + name + suffix +'.h5'
        getKerasModel(nVars, modelName, nLayers, layerSize, dropValue)
        dnnOptions = '!H:!V:FilenameModel=' + modelName + ':NumEpochs=20:TriesEarlyStopping=5:BatchSize=1024:ValidationSize=30%'
        dnnName = 'DNNMuonID' + name + suffix
        factory.BookMethod(dataloader, TMVA.Types.kPyKeras, dnnName, dnnOptions + preprocessingOptions)
        print(modelName)
        print(dnnOptions + preprocessingOptions)

# Run training, test and evaluation
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()

# DropoutValue = 0
# : Evaluation results ranked by best signal efficiency and purity (area)
# : -------------------------------------------------------------------------------------------------------------------
# : DataSet       MVA                       
# : Name:         Method:          ROC-integ
# : dataset       DNNMuonIDFullScanDP0_3_300_0: 0.935
# : dataset       DNNMuonIDFullScanDP0_4_300_0: 0.935
# : dataset       DNNMuonIDFullScanDP0_2_300_0: 0.935
# : dataset       DNNMuonIDFullScanDP0_4_100_0: 0.935
# : dataset       DNNMuonIDFullScanDP0_3_200_0: 0.935
# : dataset       DNNMuonIDFullScanDP0_3_100_0: 0.935
# : dataset       DNNMuonIDFullScanDP0_2_100_0: 0.935
# : dataset       DNNMuonIDFullScanDP0_2_200_0: 0.935
# : dataset       DNNMuonIDFullScanDP0_4_200_0: 0.935
# : dataset       DNNMuonIDFullScanDP1_3_200_1: 0.935
# : dataset       DNNMuonIDFullScanDP1_2_300_1: 0.935
# : dataset       DNNMuonIDFullScanDP1_4_300_1: 0.935
# : dataset       DNNMuonIDFullScanDP1_2_200_1: 0.935
# : dataset       DNNMuonIDFullScanDP1_3_300_1: 0.935
# : dataset       DNNMuonIDFullScanDP1_4_200_1: 0.934
# : dataset       DNNMuonIDFullScanDP1_3_100_1: 0.934
# : dataset       DNNMuonIDFullScanDP1_4_100_1: 0.934
# : dataset       DNNMuonIDFullScanDP1_2_100_1: 0.934
# : dataset       DNNMuonIDFullScanDP2_4_300_2: 0.935
# : dataset       DNNMuonIDFullScanDP2_3_300_2: 0.935
# : dataset       DNNMuonIDFullScanDP2_2_300_2: 0.934
# : dataset       DNNMuonIDFullScanDP2_2_200_2: 0.934
# : dataset       DNNMuonIDFullScanDP2_4_200_2: 0.934
# : dataset       DNNMuonIDFullScanDP2_3_200_2: 0.934
# : dataset       DNNMuonIDFullScanDP2_2_100_2: 0.934
# : dataset       DNNMuonIDFullScanDP2_4_100_2: 0.934
# : dataset       DNNMuonIDFullScanDP2_3_100_2: 0.934
# : -------------------------------------------------------------------------------------------------------------------
# : 
# : Testing efficiency compared to training efficiency (overtraining check)
# : -------------------------------------------------------------------------------------------------------------------
# : DataSet              MVA              Signal efficiency: from test sample (from training sample) 
# : Name:                Method:          @B=0.01             @B=0.10            @B=0.30   
# : -------------------------------------------------------------------------------------------------------------------
# : dataset              DNNMuonIDFullScanDP0_3_300_0: 0.271 (0.286)       0.778 (0.782)      0.975 (0.976)
# : dataset              DNNMuonIDFullScanDP0_4_300_0: 0.274 (0.285)       0.778 (0.783)      0.975 (0.976)
# : dataset              DNNMuonIDFullScanDP0_2_300_0: 0.277 (0.283)       0.777 (0.780)      0.975 (0.976)
# : dataset              DNNMuonIDFullScanDP0_4_100_0: 0.276 (0.281)       0.776 (0.779)      0.974 (0.975)
# : dataset              DNNMuonIDFullScanDP0_3_200_0: 0.271 (0.283)       0.775 (0.785)      0.975 (0.976)
# : dataset              DNNMuonIDFullScanDP0_3_100_0: 0.277 (0.277)       0.776 (0.778)      0.974 (0.975)
# : dataset              DNNMuonIDFullScanDP0_2_100_0: 0.278 (0.278)       0.776 (0.777)      0.974 (0.974)
# : dataset              DNNMuonIDFullScanDP0_2_200_0: 0.273 (0.281)       0.776 (0.782)      0.975 (0.976)
# : dataset              DNNMuonIDFullScanDP0_4_200_0: 0.275 (0.273)       0.777 (0.778)      0.974 (0.975)
# : dataset              DNNMuonIDFullScanDP1_3_200_1: 0.273 (0.279)       0.776 (0.779)      0.974 (0.975)
# : dataset              DNNMuonIDFullScanDP1_2_300_1: 0.274 (0.279)       0.777 (0.780)      0.975 (0.975)
# : dataset              DNNMuonIDFullScanDP1_4_300_1: 0.267 (0.284)       0.776 (0.783)      0.975 (0.976)
# : dataset              DNNMuonIDFullScanDP1_2_200_1: 0.275 (0.283)       0.777 (0.779)      0.974 (0.974)
# : dataset              DNNMuonIDFullScanDP1_3_300_1: 0.272 (0.281)       0.775 (0.780)      0.975 (0.975)
# : dataset              DNNMuonIDFullScanDP1_4_200_1: 0.268 (0.278)       0.775 (0.780)      0.974 (0.975)
# : dataset              DNNMuonIDFullScanDP1_3_100_1: 0.273 (0.277)       0.776 (0.775)      0.973 (0.974)
# : dataset              DNNMuonIDFullScanDP1_4_100_1: 0.274 (0.278)       0.776 (0.776)      0.973 (0.974)
# : dataset              DNNMuonIDFullScanDP1_2_100_1: 0.277 (0.276)       0.776 (0.776)      0.973 (0.973)
# : dataset              DNNMuonIDFullScanDP2_4_300_2: 0.278 (0.280)       0.777 (0.779)      0.975 (0.975)
# : dataset              DNNMuonIDFullScanDP2_3_300_2: 0.274 (0.278)       0.777 (0.778)      0.974 (0.975)
# : dataset              DNNMuonIDFullScanDP2_2_300_2: 0.268 (0.272)       0.778 (0.779)      0.974 (0.975)
# : dataset              DNNMuonIDFullScanDP2_2_200_2: 0.276 (0.277)       0.776 (0.777)      0.974 (0.974)
# : dataset              DNNMuonIDFullScanDP2_4_200_2: 0.276 (0.277)       0.776 (0.777)      0.973 (0.974)
# : dataset              DNNMuonIDFullScanDP2_3_200_2: 0.270 (0.274)       0.775 (0.776)      0.974 (0.974)
# : dataset              DNNMuonIDFullScanDP2_2_100_2: 0.273 (0.273)       0.775 (0.774)      0.972 (0.973)
# : dataset              DNNMuonIDFullScanDP2_4_100_2: 0.275 (0.273)       0.774 (0.773)      0.973 (0.973)
# : dataset              DNNMuonIDFullScanDP2_3_100_2: 0.275 (0.271)       0.773 (0.774)      0.972 (0.972)








# : Evaluation results ranked by best signal efficiency and purity (area)
# : -------------------------------------------------------------------------------------------------------------------
# : DataSet       MVA                       
# : Name:         Method:          ROC-integ
# : dataset       DNNMuonIDFullScanDP0_fine_3_100_0: 0.933
# : dataset       DNNMuonIDFullScanDP0_fine_2_200_0: 0.933
# : dataset       DNNMuonIDFullScanDP0_fine_2_100_0: 0.933
# : dataset       DNNMuonIDFullScanDP0_fine_3_200_0: 0.933
# : -------------------------------------------------------------------------------------------------------------------
# : 
# : Testing efficiency compared to training efficiency (overtraining check)
# : -------------------------------------------------------------------------------------------------------------------
# : DataSet              MVA              Signal efficiency: from test sample (from training sample) 
# : Name:                Method:          @B=0.01             @B=0.10            @B=0.30   
# : -------------------------------------------------------------------------------------------------------------------
# : dataset              DNNMuonIDFullScanDP0_fine_3_100_0: 0.267 (0.279)       0.770 (0.781)      0.974 (0.976)
# : dataset              DNNMuonIDFullScanDP0_fine_2_200_0: 0.265 (0.280)       0.769 (0.782)      0.974 (0.976)
# : dataset              DNNMuonIDFullScanDP0_fine_2_100_0: 0.268 (0.277)       0.769 (0.778)      0.974 (0.975)
# : dataset              DNNMuonIDFullScanDP0_fine_3_200_0: 0.261 (0.275)       0.768 (0.782)      0.975 (0.976)
# : -------------------------------------------------------------------------------------------------------------------

# : DataSet       MVA                       
# : Name:         Method:          ROC-integ
# : dataset       DNNMuonIDFullScanDP1_fine_3_200_1: 0.933
# : dataset       DNNMuonIDFullScanDP1_fine_2_200_1: 0.933
# : dataset       DNNMuonIDFullScanDP1_fine_2_100_1: 0.933
# : dataset       DNNMuonIDFullScanDP1_fine_3_100_1: 0.933
# : -------------------------------------------------------------------------------------------------------------------
# : 
# : Testing efficiency compared to training efficiency (overtraining check)
# : -------------------------------------------------------------------------------------------------------------------
# : DataSet              MVA              Signal efficiency: from test sample (from training sample) 
# : Name:                Method:          @B=0.01             @B=0.10            @B=0.30   
# : -------------------------------------------------------------------------------------------------------------------
# : dataset              DNNMuonIDFullScanDP1_fine_3_200_1: 0.264 (0.274)       0.770 (0.779)      0.974 (0.975)
# : dataset              DNNMuonIDFullScanDP1_fine_2_200_1: 0.265 (0.275)       0.769 (0.778)      0.974 (0.975)
# : dataset              DNNMuonIDFullScanDP1_fine_2_100_1: 0.267 (0.276)       0.769 (0.776)      0.973 (0.973)
# : dataset              DNNMuonIDFullScanDP1_fine_3_100_1: 0.262 (0.270)       0.768 (0.774)      0.973 (0.974)
# : -------------------------------------------------------------------------------------------------------------------


# : -------------------------------------------------------------------------------------------------------------------
# : DataSet       MVA                       
# : Name:         Method:          ROC-integ
# : dataset       DNNMuonIDFullScanDP2_fine_3_200_2: 0.933
# : dataset       DNNMuonIDFullScanDP2_fine_2_200_2: 0.933
# : dataset       DNNMuonIDFullScanDP2_fine_3_100_2: 0.932
# : dataset       DNNMuonIDFullScanDP2_fine_2_100_2: 0.932
# : -------------------------------------------------------------------------------------------------------------------
# : 
# : Testing efficiency compared to training efficiency (overtraining check)
# : -------------------------------------------------------------------------------------------------------------------
# : DataSet              MVA              Signal efficiency: from test sample (from training sample) 
# : Name:                Method:          @B=0.01             @B=0.10            @B=0.30   
# : -------------------------------------------------------------------------------------------------------------------
# : dataset              DNNMuonIDFullScanDP2_fine_3_200_2: 0.266 (0.275)       0.768 (0.777)      0.974 (0.974)
# : dataset              DNNMuonIDFullScanDP2_fine_2_200_2: 0.265 (0.271)       0.769 (0.774)      0.973 (0.973)
# : dataset              DNNMuonIDFullScanDP2_fine_3_100_2: 0.265 (0.270)       0.768 (0.775)      0.972 (0.973)
# : dataset              DNNMuonIDFullScanDP2_fine_2_100_2: 0.265 (0.270)       0.767 (0.774)      0.972 (0.972)
# : -------------------------------------------------------------------------------------------------------------------