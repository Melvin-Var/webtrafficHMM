import numpy as np
import time
import matlab
import os


def Evaluation(X, predProb, model, eng, outputPath):
    """
    yields FDR-controlled, standardized-scores for the data
    :param X: data to be scored (n x number Features)
    :param predProb: gives the probabilities of being in each state for each observation in X
    :param model: final, fitted HMM model
    :param eng: the MATLAB engine used for scoring
    :param outputPath: directory where to save the evaluation output
    :return: n x 1 vector for FDR-controlled, standardized scores
    """

    n = len(X)
    nComp = model.n_components
    nMix = model.n_mix
    nFeat = len(X[0])
    scoreVec = np.full(n, 0.)

    start = time.time()
    probStateMix = np.full((nComp, nMix), -1.)
    tailProbStateMix = np.full((nComp, nMix, n), -1.)
    for k in range(nComp):
        for j in range(nMix):
            probStateMix[k, j] = model.weights_[k][j]
            muKJ = np.transpose(model.means_[k][j])
            covarKJ = model.covars_[k][j]
            mixStatePvec = np.array(eng.getlocaltailp(matlab.double(X.tolist()), matlab.double(muKJ.tolist()),
                                                      matlab.double(covarKJ.tolist()), nargout=1))
            mixStatePvec = [mixStatePvec[x, 0] for x in range(n)]
            tailProbStateMix[k, j] = mixStatePvec

    print("time elapsed: %s" % (time.time() - start))

    start = time.time()
    for i in range(n):
        for k in range(nComp):
            for j in range(nMix):
                scoreVec[i] += tailProbStateMix[k, j, i] * probStateMix[k, j] * predProb[i][k]

    print("time elapsed: %s" % (time.time() - start))

    # cutPr = -2 * np.log(cutPr)
    scoreVecForSSI = np.array([-2 * np.log(max(x, np.finfo(float).tiny)) for x in scoreVec])
    ssiScore = np.array([x for x in eng.standardizebyfdrdependent(matlab.double(scoreVecForSSI.tolist()), 'score',
                                                                  'positive', 0, 1, nargout=1)]).ravel()

    # highOut = np.array([1 - x for x in scoreVec])
    # ssiScore = np.array([x for x in eng.standardizebyfdrdependent(matlab.double(highOut.tolist()), 'probabilities',
    #                                                              'unitInterval', 0, 1, nargout=1)]).ravel()

    # ssiScore = np.array([x for x in eng.standardizebyfdrdependent(matlab.double(highOut.tolist()), 'score',
    #                                                              'positive', 0, 1, nargout=1)]).ravel()

    print([np.min(ssiScore), np.max(ssiScore)])

    predClass = np.full(n, -9.)
    predMix = np.full(n, -9.)
    predSD = np.full(n, -9.)
    uncertaintyClass = np.full(n, -9.)
    for i in range(n):
        classProbsI = predProb[i, :]
        predClass[i] = classProbsI.argmax()

        mixProbs = tailProbStateMix[int(predClass[i]), :, i]
        predMix[i] = mixProbs.argmax()
        sds = [np.sqrt(model.covars_[int(predClass[i])][int(predMix[i])][x, x]) for x in range(nFeat)]
        predSD[i] = np.min(sds)

        classProbsI = list(classProbsI)
        classProbsI.remove(max(classProbsI))
        uncertaintyClass[i] = (max(predProb[i, :]) - max(classProbsI)) / max(predProb[i, :])

    augMat = np.vstack((scoreVec, ssiScore, predClass, predMix, predSD, uncertaintyClass)).T

    outputPath += "fileComm/"
    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)
    outputPath += "evaluation/"
    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)
    augPath = outputPath + "augmentation.csv"

    np.savetxt(augPath, augMat, delimiter=",")

    # return scoreVec, ssiScore, predClass, predMix, predSD
