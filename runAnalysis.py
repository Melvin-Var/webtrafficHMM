import matlab.engine
import glob
import pandas as pd
import numpy as np
#from GMMHMM import *
from featureMapping import *
from training import *
from modelEvaluation import *
from modelPlotting import *
import sys
import pickle
import csv


inputPath = "/home/d869321/CyberResearch-Data/proxyFull/part*.csv"
entityAddresses = glob.glob(inputPath)

outputPath = "/home/d869321/Presentations/hmmProxy/fullProxyMod5/"
listFeat = ["sum_request_bytes", "sum_response_bytes"]
zeroFeat = ["sum_response_bytes"]
IdentifierStr = ["/part-", "00000-", "-"]

#################################################### Set Parameters ####################################################
# trimming parameters
# effectively places the zero Response Bytes 0.22 (= log(0.8)) units below the smallest non-zero Response Bytes
ratioMin = 0.8
tailProb = 0.25

# model complexity parameters
nCompPerm = 12  # Need to change colour scheme for states if we wish to increase this
nMix = 8
extraState = 1
extraMix = 4
indContVar = [0, 1]

# HDR parameters
threshVar = 1e-04
threshDist = 0.1
minObsMode = 30
minVar = 0.0001  # this is for the covariance regularization
maxCorr = 0.9
gmmComp = 12

# Enforcement parameters
numIterSmart = 2
minCoverage = 14 / 15.

# Presentation parameters
plotFlag = True
ypos = 2.2  # tells Emission Plotter where exactly to place the titles for emission plots
cutPr = 0.001  # probability below which we raise a minor detection

HDRs = dict([("threshVar", threshVar), ("threshDist", threshDist), ("minObsMode", minObsMode), ("minVar", minVar),
             ("minCoverage", minCoverage), ("maxCorr", maxCorr), ("gmmComp", gmmComp)])

plotPar = dict([("ypos", ypos), ("resol", 0.02), ("nGrid", 400) , ("cutPr", cutPr)])

########################################################################################################################

eng = matlab.engine.start_matlab()
eng.addpath(eng.genpath("/home/d869321/CyberResearch-master/Matlab"))

start_time = time.time()
index = 30

for i in range(index - 1):
    entityAddresses.pop(0)

size = []
fileInt = index
for entityAddress in entityAddresses:

    featureMapper(entityAddress, listFeat, zeroFeat, ratioMin, tailProb, outputPath, True, IdentifierStr)

    #################### Read in feature Mapper output ####################
    featurePath = outputPath + "fileComm/featureMapper/"
    dataOrg = np.loadtxt(open(featurePath + "dataMat.csv", "rb"), delimiter=",", skiprows=0)
    dataTrim = np.loadtxt(open(featurePath + "dataTrimMat.csv", "rb"), delimiter=",", skiprows=0)
    dfContext = pd.read_csv(featurePath + "dataExtended.csv", sep='\t')
    with open(featurePath + "identifier.csv", 'r') as f:
        reader = csv.reader(f, dialect='excel')
        identifier_list = list(reader)

    identifier = identifier_list[0][0]
    entity = identifier_list[0][1]
    #.....................................................................#

    size = size + [dataOrg.shape[0]]

    n = dataOrg.shape[0]
    maxState = int(np.floor(n / (3 * 8 * 8)))
    nComp = min(nCompPerm, maxState)

    structHMM = dict([("nComp", nComp), ("nMix", nMix), ("extraState", extraState), ("extraMix", extraMix),
                      ('indContVar', indContVar)])

    print(['~~~~~~~~~~', fileInt, identifier, dataOrg.shape, nComp, '~~~~~~~~~~'])

    fitGMMHMM(numIterSmart, tailProb, dataOrg, dataTrim, structHMM, HDRs, outputPath)

    #################### Read in training output ####################
    jointModel = pickle.load(open(outputPath + "/fileComm/training/model.sav", 'rb'))
    distModes = np.loadtxt(open(outputPath + "/fileComm/training/distModes.csv", "rb"), delimiter=",", skiprows=0)
    predZ = np.loadtxt(open(outputPath + "/fileComm/training/predZ.csv", "rb"), delimiter=",", skiprows=0)
    predProb = np.loadtxt(open(outputPath + "/fileComm/training/predProb.csv", "rb"), delimiter=",", skiprows=0)

    with open(outputPath + "/fileComm/training/modelStateMix.csv", 'r') as f:
        reader = csv.reader(f, dialect='excel')
        model_list = list(reader)

    readDict = csv.DictReader(open(outputPath + "/fileComm/training/fitStats.csv"))
    for row in readDict:
        fitStats = row

    nCompUse1 = int(model_list[0][0])
    nCompUse3 = int(model_list[0][1])
    nMixUse1 = int(model_list[0][2])
    nMixUse3 = int(model_list[0][3])

    bestScore = float(fitStats['bestScore'])
    firstScore = float(fitStats['firstScore'])
    firstIterate = float(fitStats['firstIterate'])
    smartIterateVec = [float(x) for x in fitStats['smartIterateVec'].replace('[','').replace(']','').split()]
    secondIterateVec = [float(x) for x in fitStats['secondIterateVec'].replace('[','').replace(']','').split()]
    secondEnforceScore = float(fitStats['secondEnforceScore'])
    typeFit = fitStats['typeFit']
    secondEnforceScoreIterate = float(fitStats['secondEnforceScoreIterate'])
    bestCoverage = float(fitStats['bestCoverage'])
    bestSim = float(fitStats['bestSim'])
    bestJS = float(fitStats['bestJS'])
    # ..............................................................#

    numPar = (nCompUse3 * nCompUse3) + (3 * nCompUse3 * nMixUse3) - nCompUse3 + 1
    line = str(
        [fileInt, identifier, typeFit, [nCompUse1, nMixUse1], [nCompUse3, nMixUse3, numPar], firstScore, firstIterate,
         smartIterateVec,
         secondIterateVec, [secondEnforceScore, secondEnforceScoreIterate], [bestCoverage, bestSim, bestJS],
         dataOrg.shape[0]])
    print(line)

    f = open(outputPath + "yScores.csv", 'a')
    f.write(line + '\n')
    f.close()

    simulFile = outputPath + "simul_" + str(fileInt) + "_" + str(entity) + ".csv"
    np.savetxt(simulFile, np.asmatrix(distModes), delimiter=",")

    transMatFile = outputPath + "trans_" + str(fileInt) + "_" + str(entity) + ".csv"
    np.savetxt(transMatFile, jointModel.transmat_, delimiter=",")
    transMat = np.savetxt(sys.stdout, jointModel.transmat_, '%5.3f')
    print(transMat)

    fileStr = str(fileInt) + "_" + str(entity)

    Evaluation(dataOrg, predProb, jointModel, eng, outputPath)


    #################### Read in evaluation output ####################
    augMat = np.loadtxt(open(outputPath + "/fileComm/evaluation/augmentation.csv", "rb"), delimiter=",", skiprows=0)
    dfContext['predClass'] = augMat[:, 2]
    dfContext['predMix'] = augMat[:, 3]
    dfContext['predSD'] = augMat[:, 4]
    dfContext['tailprob'] = augMat[:, 0]
    dfContext['ssiScore'] = augMat[:, 1]
    dfContext['uncertainty'] = augMat[:, 5]
    cols = dfContext.columns.tolist()
    cols = cols[-6:] + cols[1:-6]
    dfContext = dfContext[cols]
    #.................................................................#

    if plotFlag:
        HMMplotting(dataOrg, jointModel, predZ, plotPar, dfContext['tailprob'], dfContext['ssiScore'],
                    dfContext['predClass'], outputPath, ['log(Req.Byte)', 'log(Resp.Byte)'], fileStr, bestScore, typeFit)


    workDirContext = outputPath + "data" + str(fileInt) + "_" + str(entity) + "_" + str(identifier) + ".csv"
    dfContext.to_csv(workDirContext, sep='\t')

    pickle.dump(jointModel, open(outputPath + "model" + str(fileInt) + "_" + str(identifier) + ".sav", 'wb'))

    fileInt += 1

print([min(size), max(size)])
elapsed_time = time.time() - start_time
print("Total Time elapsed: %s seconds" % elapsed_time)
