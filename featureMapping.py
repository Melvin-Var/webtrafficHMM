import numpy as np
import csv
import pandas as pd
import copy
import os


def trimDF(df, tailProb):
    """
    Trims all points above and below the tailProb & 1-tailProb percentiles to the thresholds
    :param df: dataframe to be trimmed
    :param tailProb: the lowest and highest tailProb % of data is trimmed to the relevant thresholds
    :return: the trimmed Data frame
    """

    lowThresh = np.percentile(df, tailProb, axis=0)
    highThresh = np.percentile(df, 100 - tailProb, axis=0)

    df = np.maximum(df, lowThresh)
    df = np.minimum(df, highThresh)

    return df


def featureMapper(entityAddr, listFeat, zeroFeat, ratioMin, tailProb, outputPath, identifier, IdentifierStr):
    """
    reads the required features from the relevant csv for the entity
    :param entityAddr: the filepath to the csv for the entity
    :param listFeat: list of strings describing the features to be processed by GMMHMM
    :param zeroFeat: list of strings describing the features that sometimes have zero values
    :param ratioMin: fraction describing how to shift zero-valued obs in relation to smallest positive values
    :param tailProb: what percentage of data to trim for placement in dataTrim
    :param outputPath: where to save csv files to
    :param identifier: Boolean of whether we want unique identifier for each entity
    :param IdentifierStr: list of 3 strings that determine the identifier within csv filename #["/part-","00000-", "-"]
    :return: N/A - All output from this function is saved to files in particular directory
    """

    d = len(listFeat)

    if identifier:
        try:
            entity = entityAddr.split(IdentifierStr[0])[1]
            identifier = entity.split(IdentifierStr[1])[1].split(IdentifierStr[2])[0]
        except:
            identifier = "unknown"
    else:
        identifier = "unknown"

    df = pd.read_csv(entityAddr)
    dfContext = copy.deepcopy(df)

    for i in range(d):

        if listFeat[i] in zeroFeat:
            # make zero bytes equal to some higher value
            minFeat = min(df[df[listFeat[i]] > 0][listFeat[i]])
            df[listFeat[i]] = np.maximum(df[listFeat[i]], minFeat * ratioMin)

        df[listFeat[i]] = df[listFeat[i]].apply(lambda x: np.log(x))

        entity = df['entity'][0]  # this is an alternative identifier

        means = df.mean(axis=0)
        stds = df.std(axis=0)

        featInd = list(df).index(listFeat[i])  # gives the column index that has the i-th Feature in listFeat
        df[listFeat[i]] = df[listFeat[i]].apply(lambda x: (x - means[featInd]) / stds[featInd])

    dfReduced = pd.concat([df[listFeat[x]] for x in range(d)], axis=1)
    dfTrim = trimDF(dfReduced, tailProb)

    dataMat = dfReduced[[listFeat[x] for x in range(d)]].as_matrix()
    dataMatTrim = dfTrim[[listFeat[x] for x in range(d)]].as_matrix()

    outputPath += "fileComm/"
    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)
    outputPath += "featureMapper/"
    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)
    dataPath = outputPath + "dataMat.csv"
    dataTrimPath = outputPath + "dataTrimMat.csv"
    dfPath = outputPath + "dataExtended.csv"
    identPath = outputPath + "identifier.csv"

    np.savetxt(dataPath, dataMat, delimiter=",")
    np.savetxt(dataTrimPath, dataMatTrim, delimiter=",")
    dfContext.to_csv(dfPath, sep='\t')

    with open(identPath, 'w') as resultFile:
        wr = csv.writer(resultFile, dialect='excel')
        wr.writerow([identifier, entity])
