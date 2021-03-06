import numpy as np
import copy
from scipy.stats import multivariate_normal
from scipy import spatial
from scipy import linalg
from sklearn.mixture import GaussianMixture
import time
import multiprocessing as mp
from matplotlib import pyplot as plt
from hmmlearn import hmm
from sklearn import cluster
import warnings
import sys
import os
import csv
import pickle


def matrixCreator(nComp, stayPr):
    """
    creates a strongly diagonal transition matrix with "stayPr" on diagonals and "1-stayP" spread among off-diagonals
    :param nComp: how many states
    :param stayPr: probability along diagonals
    :return: nComp x nComp strongly diagonal transition matrix
    """

    transmat = np.full((nComp, nComp), -1.)

    # for states outside subset, we split 1-stayPr amongst off-diagonal nodes
    offProb = round((1 - stayPr) / (nComp - 1), 10)

    for i in range(nComp):
        for j in range(nComp):
            if i != j:
                transmat[i, j] = offProb
            else:
                transmat[i, j] = stayPr

    return transmat


def trimStates(nComp, X, predZ, tailProb):
    """
    For all points predicted to lie in State k, points below/above the tailProb & 1 - tailProb percentiles are trimmed
    :param nComp: number of HMM states
    :param X: the data (n x nFeat)
    :param predZ: the predicted states for each of the n observations
    :param tailProb: at what level should we trim each of the states
    :return: a state-based trimmed version of X
    """

    trimX = copy.deepcopy(X)
    nFeat = len(X[0])
    n = len(X)

    for j in range(nFeat):
        #for i in range(nComp):
        for i in np.unique(predZ):  # avoids states which don't have a single observation predicted to lie in them

            subsetBytes = [x[j] for x in X]
            isI = predZ == i  # vector of Boolean that flags True if predicted state is i
            subsetBytes = [x for x, y in zip(subsetBytes, isI) if y]
            indState = [x for x, y in zip(range(n), isI) if y]

            try:
                lowThresh = np.percentile(subsetBytes, tailProb)
                highThresh = np.percentile(subsetBytes, 100 - tailProb)

                for k in range(len(subsetBytes)):
                    trimX[indState[k], j] = max(min(X[indState[k], j], highThresh), lowThresh)

            except BaseException as e:
                print("Number of observations predicted to lie in %s-th state is %s." % (i, len(subsetBytes)))
                print("Error caused: " + str(e))

    return trimX


def klGaussiansMoments(mu1, mu2, cov1, cov2):
    """
    Calculates the Kullback Leibler divergence between two Gaussian distributions
    :param mu1: mean of first distribution
    :param mu2: mean of second distribution
    :param cov1: covariance of first distribution
    :param cov2: covariance of second distribution
    :return: the KL divergence between the two distributions
    """

    d = len(mu1)

    term1 = np.log(np.linalg.det(cov2) / np.linalg.det(cov1)) - d
    term2 = np.trace(np.matmul(np.linalg.inv(cov2), cov1))
    term3 = np.matmul(np.matmul((mu2[np.newaxis] - mu1[np.newaxis]), np.linalg.inv(cov2)),
                      (mu2[np.newaxis] - mu1[np.newaxis]).T)[0][0]

    return 0.5 * (term1 + term2 + term3)


def jsGaussians(model, state1, state2, mix1, mix2, nSim):
    """
    Calculate the Jensen-Shannon distance (= sqrt(Jensen-Shannon divergence))
    :param model: the HMM model where each state-mixture is Gaussian (i.e. it is a Gaussian Mixture HMM)
    :param state1: corresponding state for first distribution
    :param state2: corresponding state for second distribution
    :param mix1: corresponding mixture for first distribution
    :param mix2: corresponding mixture for second distribution
    :param nSim: how many simulations to use to calculate the Jensen-Shannon distance
    :return: Jensen-Shannon distance between the two relevant Guassians corresponding to the appropriate state-mixtures
    """

    cov1 = model.covars_[state1][mix1]
    cov2 = model.covars_[state2][mix2]
    mu1 = model.means_[state1][mix1]
    mu2 = model.means_[state2][mix2]

    X = multivariate_normal.rvs(size=nSim, mean=mu1, cov=cov1)
    ln_p_X = multivariate_normal.logpdf(X, mean=mu1, cov=cov1)
    ln_q_X = multivariate_normal.logpdf(X, mean=mu2, cov=cov2)
    ln_mix_X = np.logaddexp(ln_p_X, ln_q_X)

    Y = multivariate_normal.rvs(size=nSim, mean=mu2, cov=cov2)
    ln_p_Y = multivariate_normal.logpdf(Y, mean=mu1, cov=cov1)
    ln_q_Y = multivariate_normal.logpdf(Y, mean=mu2, cov=cov2)
    ln_mix_Y = np.logaddexp(ln_p_Y, ln_q_Y)

    dist = (ln_p_X.mean() - (ln_mix_X.mean() - np.log(2)) + ln_q_Y.mean() - (ln_mix_Y.mean() - np.log(2))) / 2
    dist = min(max(dist, 0), 1)  # sometimes the distance is very slightly negative

    return np.sqrt(dist)


def GMMfit(listGMM):
    """
    thread-friendly process that fits the GMM to the multivariate data
    :param listGMM: list: [data, random_seed, n_components, n_init]
    :return: list: [gmm model score, gmm model]
    """

    jointData = listGMM[0]

    # sometime GMM will get create a class whose size is less than number of components, raising an error.
    # In such cases, we change the seed and retry fitting the gmm with less components
    iter = 0
    successGMM = False
    while not successGMM:
        try:
            gmm = GaussianMixture(n_components=(listGMM[2] - iter), n_init=listGMM[3], random_state=(listGMM[1] + iter))
            gmm = gmm.fit(jointData)
            successGMM = True
        except:
            print("GMM not successful. Trying again.")
            iter += 1

    score = gmm.score(jointData)

    return [score, gmm]


def fitGMM(jointData, nInit, nComp, HDRstate, threshFreq, threshVar, calcDist, indContVar, plotter=False, mulThr=False):
    """
    finds the best fitting GMM (with nComp mixtures) among the nInit tried. If HDRstate is true, restrict to
    corresponding High Density Regions otherwise all clusters returned. If calcDist true, calculate distance between
    relevant clusters
    :param jointData: the data (n x num Features)
    :param nInit: how many restarts of the HMM should we run in the quest to find the best model
    :param nComp: number of components for the GMM (Note: this is NOT the number of states of the HMM)
    :param HDRstate: boolean of whether or not we want the states to correspond to high density regions (HDRs)
    :param threshFreq: High Density regions (HDRs) must have at least threshFreq number of observations assigned to it
    :param threshVar: High Density regions (HDRs) must have at most variance threshVar in its corresponding mixture
    :param calcDist: boolean for whether or not we should calculate the distance between the HDRs time signatures
    :param indContVar: indices of the continuous variables. Identify High Density Regions based just on such variables.
    :param plotter: boolean indicating whether or not we should plot the GMMs predictions over the data
    :return: list: [gmm model, weights of HDR's mixtures, means of HDR's mixtures, covars of HDRs's,
                    distance between the time signatures of the HDRs]
    """

    start = time.time()
    numThreads = 4
    seed = np.random.randint(1, 10000, numThreads)  # generates numThreads number of seeds
    nThr = int(np.floor(nInit / numThreads))  # insists on the number of fits of GMM being a multiple of numThreads

    nIter = 0
    threshFreqUse = threshFreq
    threshVarUse = threshVar
    goodComps = False  # in the case of finding HDRs, it is ideal hat at least three HDRs are found
    while not goodComps:

        if mulThr:
            # Attempt to ensure good fit by fitting the GMM multiple times. Speed this by running GMM models over 4 threads
            pool = mp.Pool()
            results = list(pool.imap_unordered
                           (GMMfit, [[jointData, seed[0], nComp, nThr],
                                     [jointData, seed[1], nComp, nThr],
                                     [jointData, seed[2], nComp, nThr],
                                     [jointData, seed[3], nComp, nThr]]))
            pool.close()
            pool.join()

            # find the best fitting GMM returned by the 4 threads
            scores = np.array([x[0] for x in results])
            bestGMMind = np.argmax(scores)
            gmm = results[bestGMMind][1]

        else:

            scores = np.full(numThreads, -1.)
            GMMmodels = [-1 for x in range(numThreads)]

            for i in range(numThreads):
                iter = 0
                successGMM = False
                while not successGMM:
                    try:
                        gmm = GaussianMixture(n_components=(nComp - iter), n_init=nThr,
                                              random_state=(seed[i] + iter))
                        gmm = gmm.fit(jointData)
                        successGMM = True
                    except:
                        print("GMM not successful. Trying again.")
                        iter += 1

                scores[i] = gmm.score(jointData)
                GMMmodels[i] = gmm

            labels = gmm.predict(jointData)
            bestGMMind = np.argmax(scores)
            gmm = GMMmodels[bestGMMind]

        # for each of the components, finds the minimum variance across the nFeat features
        spread = np.array([min([gmm.covariances_[x][y][y] for y in indContVar]) for x in range(nComp)])

        if HDRstate:
            # find the weights, means and covariances of the Gaussian components corresponding to high density regions
            weights = gmm.weights_[(gmm.weights_ > threshFreqUse) & (spread < threshVarUse)]
            means = gmm.means_[(gmm.weights_ > threshFreqUse) & (spread < threshVarUse)]
            covars = gmm.covariances_[(gmm.weights_ > threshFreqUse) & (spread < threshVarUse)]
        else:
            weights = gmm.weights_
            means = gmm.means_
            covars = gmm.covariances_

        # sort the means & covariances in descending order of component weights
        try:
            means = np.array([x for _, x in sorted(zip(weights, means), key=lambda pair: pair[0], reverse=True)])
        except:
            print("------")
            print(weights)
            print(means)
            print(zip(weights, means))
            print(sorted(zip(weights, means), reverse=True))
            print([len(weights), means.shape])
            print("******")
        covars = np.array([x for _, x in sorted(zip(weights, covars), key=lambda pair: pair[0], reverse=True)])

        label = np.array(range(nComp))
        if HDRstate:
            # find the labels of components that have high density
            labelStates = label[(gmm.weights_ > threshFreqUse) & (spread < threshVarUse)]
            labelStates = np.array([x for _, x in sorted(zip(weights, labelStates), reverse=True)])
            if (len(labelStates) < 3) and calcDist:
                threshFreqUse *= 0.9
                threshVarUse *= 1.5
                nIter += 1
                if nIter == 3:
                    threshVarUse = 0.5
                else:
                    if nIter == 5:
                        print("Only found %s HDRs after two iterations" % len(labelStates))
                        goodComps = True
            else:
                goodComps = True
        else:
            labelStates = np.array([x for _, x in sorted(zip(weights, label), reverse=True)])
            goodComps = True

    nMode = len(labelStates)

    weights = sorted(weights, reverse=True)

    distModes = np.full((nMode, nMode), 0.)

    if calcDist:

        # calculate the times for each of the high density states
        ind = range(len(jointData))  # yields [0, 1, ..., nObs - 1]
        predStatesInd = []
        for i in labelStates:
            # Recall "labels" is predicted label by GMM for each of nObs observations
            isI = labels == i  # vector of Boolean that flags True if predicted state is i.
            subsetInd = [x for x, y in zip(ind, isI) if y]

            predStatesInd = predStatesInd + [subsetInd]

        flattenInds = [item for sublist in predStatesInd for item in sublist]
        stdDev = np.std(flattenInds)
        binWidth = round(1.75 * stdDev * (len(flattenInds) ** (-1 / 3)))
        # we make the bins just cover the time grid [1, 2,..., n] - assuming HDRs mostly span entire interval
        nBins = round(len(jointData) / binWidth)
        binWidth = np.ceil(len(jointData) / nBins)

        print("No bin: %s" % nBins)

        # create bins (each of size binWidth) so that they span the time window [0, nObs]
        upBd = (np.ceil(len(labels) / binWidth) * binWidth)
        nBins = int(upBd / binWidth)
        bins = [int(x * binWidth) for x in range(nBins + 1)]

        indFreq = np.full((nBins, nMode), 0.)

        # calculate proportions of a particular high density region in each of the time bins
        for i in range(nMode):
            digitized = np.digitize(predStatesInd[i], bins) - 1  # subtract 1 so lying in first bin equivalent to = 0
            yOld = np.bincount(digitized)
            ii = np.nonzero(yOld)[0]

            # y's stop recording frequencies after max obs value
            y = np.array(list(yOld) + [0 for x in range(nBins - max(ii) - 1)])
            # change these to proportions in order to more readily compare various labels
            y = np.array([x / len(predStatesInd[i]) for x in y])
            iii = range(nBins + 1)

            try:
                indFreq[:, i] = y
            except:
                print(np.vstack((ii, yOld[ii])).T)
                print(np.vstack((iii, y[iii])).T)

        # calculate distance between high density regions based on their relative activity across time
        for i in range(nMode):
            for j in range(nMode):
                distModes[i, j] = round(spatial.distance.cosine(indFreq[:, i], indFreq[:, j]), 3)

    # plotter only works well if only 2 features (developed assuming we are dealing with Response/Request bytes)
    if plotter:
        print("Elapsed time: %s" % (time.time() - start))
        print(weights)
        print(means)
        print(covars)

        labels = gmm.predict(jointData)
        colours = [np.log(min(gmm.covariances_[labels[x]][0][0], gmm.covariances_[labels[x]][1][1]))
                   for x in range(len(labels))]
        plt.scatter(jointData[:, 0], jointData[:, 1], c=colours, s=0.1, cmap='viridis')
        plt.title("Response/Request bytes for user, colour coded by log(Min Variance of Mixture)")
        plt.xlabel("log(Request Bytes)")
        plt.ylabel("log(Response Bytes)")
        plt.colorbar()
        plt.show()

    return gmm, weights, means, covars, distModes


def covarRegularizer(model, minVar, maxCorr):
    """
    regularize the covariances of the Gaussian mixture components of the HMMs
    :param model: the fitted HMM
    :param minVar: lower bound enforced for the variance of all the mixtures
    :param maxCorr: maximum allowed correlation
    :return: modified Gaussian Mixture covariances
    """

    nComp = model.n_components
    nMix = model.n_mix
    covars = model.covars_
    dim = len(covars[0][0][0])

    for i in range(nComp):

        for j in range(nMix):

            if dim == 1:
                covars[i][j][0][0] = max(covars[i][j][0][0], minVar)
            else:

                for k in range(dim):
                    otherFeat = list(range(dim))
                    otherFeat.remove(k)
                    for l in otherFeat:
                        # sometimes fitted model yields infinite variances. In such cases, we make corr = 0
                        if max(covars[i][j][k][k], covars[i][j][l][l]) < np.inf:
                            corr = covars[i][j][k][l] / np.sqrt(covars[i][j][k][k] * covars[i][j][l][l])
                        else:
                            corr = 0

                        # in case where variances are tiny, doesn't seem to make sense to allow strong correlations
                        if min(covars[i][j][k][k], covars[i][j][l][l]) < 0.0002:
                            maxCorrUse = 0.1
                        else:
                            maxCorrUse = maxCorr

                        # correlations truncated to lie between [-maxCorrUse, maxCorrUse]
                        if corr >= 0:
                            corr = min(corr, maxCorrUse)
                        else:
                            corr = max(corr, -1 * maxCorrUse)

                        covars[i][j][k][k] = min(max(covars[i][j][k][k], minVar), 10) + 1e-5

                        covars[i][j][k][l] = corr * np.sqrt((covars[i][j][k][k] - 1e-5)
                                                            * min(max(covars[i][j][l][l], minVar), 10))

                        covars[i][j][l][k] = covars[i][j][k][l]

            covars[i][j] = (covars[i][j] + covars[i][j].T) / 2  # ensures covariance matrix is symmetric

            # occasionally fit will be successful despite some of the mixtures having covariances with nan entries.
            # in such cases, we change the covariance matrix to a diagonal matrix with large variances
            try:
                linalg.eig(covars[i, j])
            except:
                covars[i, j] = np.diag([10. for x in range(dim)])

    return covars


def startprobRegularizer(model, predZ):
    """
    regularize the starting probability vector of the HMMs
    :param model: the fitted HMM
    :param predZ: the predictions of the HMM for the data. Used to make starting prob. = relative predicted state prop.
    :return: modified starting probabilities
    """

    nComp = model.n_components
    startprob = model.startprob_
    n = len(predZ)

    for i in range(nComp):
        # find the subset of indices that are predicted to belong to state i
        subsetInd = range(n)
        isI = predZ == i  # vector of Boolean that flags True if predicted state is i
        subsetInd = [x for x, y in zip(subsetInd, isI) if y]
        startprob[i] = len(subsetInd) / n

    return startprob


def Initialization(model, X, nInitMode, nCompMode, minObsMode, threshVarMode, modeWght, threshDist, indContVar,
                   extraState, extraMix):
    """
    Initialization(jointModel, joint_bytes, 4, 50, 0.02, 5e-04, 0.9, 1000, 1. / 3, [0, 1])
    choose appropriate initializations for the means, covariances and weights for the Gaussian mixtures of the HMM
    :param model: the HMM model
    :param X: the data (n x num Features)
    :param nInitMode: how many GMMs should we run for best fitting GMM (used to find High Density Regions, HDRs)
    :param nCompMode: how many mixtures (or components) should the GMM be fitted with
    :param minObsMode: High Density regions (HDRs) must have at least minObsMode no. of observations assigned to it
    :param threshVarMode: High Density regions (HDRs) must have at most variance threshVar in its corresponding mixture
    :param modeWght: total weight assigned to mixtures tied to the mode
    :param threshDist: upper bound of distance between time signatures of HDRs. Below bound, HDRs come from same state
    :param indContVar: list of indices of variables that are continuous
    :param extraState: how many extra states should we have for dealing with behaviour outside high density regions
    :param extraMix: how many extra mixtures we need for each state to deal with values outside high density regions
    :return: means, covars, weights of the HMM's Gaussian Mixtures
    """

    threshScree = 0.01

    nMix = model.n_mix
    nComp = model.n_components
    nFeat = len(X[0])
    threshFreqMode = minObsMode / len(X)

    successInit = False
    iter = 0
    while not successInit:
        try:
            model._init(X)
            successInit = True
        except:
            # enable the model to get simpler (i.e. less states/ mixture components) in order to fit an HMM
            if np.abs(nComp - nMix) > 1:
                if nComp > nMix:
                    nComp -= 1
                else:
                    nMix -= 1
            else:
                nComp -= 1
                nMix -= 1
            if nComp <= 2 and nMix <= 1:
                iter += 1
            nComp = max(2, nComp)
            nMix = max(1, nMix)

            # we do away with the need for extra state and mixtures if we need to simplify the model
            # this prevents us from getting stuck in a loop
            if nComp < 4:
                extraState = 0
            if nMix < 5:
                extraMix = 0

            model = hmm.GMMHMM(n_components=nComp, n_mix=nMix, covariance_type="full", init_params="s")
            print([[nComp, extraState], [nMix, extraMix]])
            if iter > 1:
                raise ValueError("Cannot successfully initiate even a 2 state, 1 Gaussian mixture model")

    means = copy.deepcopy(model.means_)
    covars = copy.deepcopy(model.covars_)
    weights = copy.deepcopy(model.weights_)

    # keep looking for modes (high-density regions) until we find at least half max(nMix,nComp) modes
    enoughMode = False
    numIter = 0
    while not enoughMode:
        print("Finding HDRs")
        gmm, wghtMode, muMode, sigMode, distModes = fitGMM(X, nInitMode, nCompMode, True, threshFreqMode, threshVarMode,
                                                           True, indContVar, False)

        numMode = len(wghtMode)
        print("Found %s HDRS" % numMode)

        nIter = 1
        tooManyStates = True
        nCompMax = nComp - extraState  # we can't have more groups of HDRs than this
        while tooManyStates:

            if numMode >= 4:
                # create adjacency matrix where adjacent pts are one of the 2 closest pts to node &
                # have dist < threshSimultaneity
                simModes = np.full((numMode, numMode), 0.)
                for i in range(numMode):
                    closeInd = np.argpartition(distModes[i, :], 3)[0:3]
                    maskInd = distModes[i, closeInd] < threshDist
                    finalInd = [x for x, y in zip(closeInd, maskInd) if y]
                    for j in finalInd:
                        simModes[i, j] = 1
                        simModes[j, i] = 1

                diagonal = np.full(numMode, 0.)
                for i in range(numMode):
                    diagonal[i] = np.sum(simModes[i]) ** -0.5
                D = np.diag(diagonal)  # this variable is equal to D^-0.5 in many papers
                Lsym = np.diag(np.full(numMode, 1.)) - np.matmul(np.matmul(D, simModes), D)
                vals, vecs = np.linalg.eigh(Lsym)

                eigen_1st_diff = vals[1:numMode] - vals[0:(numMode - 1)]
                eigen_2nd_diff = eigen_1st_diff[1:(numMode - 1)] - eigen_1st_diff[0:(numMode - 2)]
                # catches case where gap never > scree gap (which would otherwise return indCut = 0)
                if max(eigen_2nd_diff) > threshScree:
                    numIndepMode = np.argmax(eigen_2nd_diff > threshScree) + 2
                else:
                    numIndepMode = numMode

                print("Finding HDR cluster")
                preds = cluster.KMeans(n_clusters=numIndepMode, init='k-means++').fit_predict(vecs[:, 1:numIndepMode])
                print("HDR clusters found")

                if numIndepMode <= nCompMax:
                    tooManyStates = False

                    cliqueInds = []
                    for i in range(max(preds + 1)):
                        subsetIndHDR = range(numMode)
                        isIhdr = preds == i  # vector of Boolean that flags True if HDR lies in cluster i
                        subsetIndHDR = [x for x, y in zip(subsetIndHDR, isIhdr) if y]
                        cliqueInds = cliqueInds + [subsetIndHDR]

                    maxNumSpike = max([len(x) for x in cliqueInds])

                else:
                    threshDist *= 1.1
                    nIter += 1

                    if nIter > 10:
                        tooManyStates = False

                        cliqueInds = []
                        for i in range(nCompMax):
                            subsetIndHDR = range(numMode)
                            isIhdr = preds == i  # vector of Boolean that flags True if HDR lies in cluster i
                            subsetIndHDR = [x for x, y in zip(subsetIndHDR, isIhdr) if y]
                            cliqueInds = cliqueInds + [subsetIndHDR]

                        maxNumSpike = max([len(x) for x in cliqueInds])
                    else:
                        print("HDRs form too many clusters (i.e. states). Making it easier for HDRs to cluster.")

            else:
                cliqueInds = [[x] for x in range(numMode)]
                numIndepMode = copy.deepcopy(numMode)
                maxNumSpike = 1
                tooManyStates = False

        print(cliqueInds)

        numIter += 1
        if numIndepMode < 3:
            threshFreqMode *= 0.9
            threshVarMode *= 1.5
            if numIter >= 3:
                if numIndepMode > 0:
                    enoughMode = True
                else:
                    raise ValueError('No high density regions found after multiple iterations.')
            else:
                print("Less than 3 states for the HDRs. Let's make it easier to find HDRs.")
        else:
            if (maxNumSpike > nMix - extraMix) or (numIndepMode > nComp - extraState):
                threshFreqMode *= 1.5
                print("HDRs use up too many Mixtures and/or States. Let's make it more difficult to find HDRs.")
            else:
                enoughMode = True

    print(distModes)
    nComp = min(nComp, numIndepMode + extraState)
    nMix = min(nMix, maxNumSpike + extraMix)
    weights = weights[:nComp, :nMix]
    means = means[:nComp, :nMix]
    covars = covars[:nComp, :nMix]

    for k in range(numIndepMode):

        numMode = len(cliqueInds[k])  # number of spikes for k-th state
        otherInd = [x + numMode for x in range(nMix - numMode)]  # indices for mix components not used for modes

        meansVec = np.full((nMix, nFeat), 0.)
        for i in range(numMode):
            meansVec[i] = muMode[cliqueInds[k][i]]
        for i in otherInd:
            meansVec[i] = means[k][i]

        means[k] = meansVec

        covarVec = np.full((nMix, nFeat, nFeat), 0.)
        for i in range(numMode):
            covarVec[i] = sigMode[cliqueInds[k][i]]
        for i in otherInd:
            covarVec[i] = (1.5 * covars[k][i])  # boost the variance slightly to enable system to adapt to nonspiky data

        covars[k] = covarVec

        weightVec = np.full(nMix, 0.)
        for i in range(numMode):
            if len(otherInd) > 0:
                weightVec[i] = modeWght * (wghtMode[cliqueInds[k][i]] / sum([wghtMode[x] for x in cliqueInds[k]]))
            else:
                weightVec[i] = 1 * (wghtMode[cliqueInds[k][i]] / sum([wghtMode[x] for x in cliqueInds[k]]))
        for i in otherInd:
            weightVec[i] = (1 - modeWght) / len(otherInd)

        weights[k] = weightVec

    weightVec = np.full(nMix, 1 / nMix)
    for k in [x + numIndepMode for x in range(nComp - numIndepMode)]:
        weights[k] = weightVec

    return nComp, nMix, means, covars, weights


def Iteration(model, X, predZ, modeWght, gmmComp, minObs, threshVar, indContVar):
    """
    iterate upon the fitted model created with the smart initialization in the hope of better emission dist. estimates
    :param model: the regularized, fitted HMM resulting from a smart initialization of the mixtures
    :param X: the data (n x number Features)
    :param predZ: the predicted states of the observations according to the model passed as argument
    :param modeWght: total weight assigned to mixtures that are tied to High Density Regions
    :param gmmComp: how many components should the GMM attempt to fit
    :param indContVar: list of indices of variables that are continuous
    :param minObs: minimum number of observations that should be found in the various High Density Regions (HDRs)
    :param threshVar: maximum variance that GMM component of a High Density region should have
    :return: mean, covars and weights of the Gaussian mixtures to be used for a new initialization
    """

    nInits = 12  # how many times to try fitting GMM to subset of data predicted to lie in state k

    nComp = model.n_components
    nMix = model.n_mix
    nFeat = len(X[0])

    means = copy.deepcopy(model.means_)
    covars = copy.deepcopy(model.covars_)
    weights = copy.deepcopy(model.weights_)

    for k in range(nComp):

        subsetBytes = X
        isI = predZ == k  # vector of Boolean that flags True if predicted state is k
        subsetBytes = np.array([x for x, y in zip(subsetBytes, isI) if y])

        if len(subsetBytes) > 0:
            threshFreq = minObs / len(subsetBytes)

            useComp = gmmComp
            gmmSuccess = False
            while not gmmSuccess:
                try:
                    gmm, wghtMode, muMode, sigMode, distModes = fitGMM(subsetBytes, nInits, useComp, True, threshFreq,
                                                                       threshVar, False, indContVar, False)
                    gmmSuccess = True
                except:
                    print("Could not Fit GMM on Iterative Step")
                    useComp -= 1

            nMode = len(wghtMode)

            if nMode >= nMix:  # we take the first nMix of weights, mean & covars & we don't leave weight for other data
                wghtMode = wghtMode[:(nMix - 1)]
                muMode = muMode[:(nMix - 1)]
                sigMode = sigMode[:(nMix - 1)]
                warnings.warn("Too many High Density Areas for observation's likely state. Try increase nMix/extraMix.")
                nMode = len(wghtMode)

            # We wish to choose which mixture components for k-th state need to be modified
            chooseMix = []
            indAvail = range(nMix)
            for i in range(nMode):
                # difference from mode for nMix components
                diffFromModeAvgs = np.array([klGaussiansMoments(means[k][x], muMode[i], covars[k][x],
                                                                sigMode[i]) for x in indAvail])

                # diffFromModeAvgs = np.array([np.linalg.norm(x - muMode[i]) for x in means[k][indAvail]])

                indMin = int(np.where(diffFromModeAvgs == diffFromModeAvgs.min())[0][0])
                possMix = indAvail[indMin]
                indAvail = np.delete(indAvail, indMin)
                chooseMix = chooseMix + [possMix]

        if (len(chooseMix) > 0) and (len(subsetBytes) > 0):
            otherInd = [-1 for x in range(nMix - nMode)]
            ind = 0
            for i in range(nMix):
                if not (i in chooseMix):
                    otherInd[ind] = i
                    ind += 1
        else:
            otherInd = range(nMix)

        meansVec = np.full((nMix, nFeat), 0.)
        for i in range(nMode):
            meansVec[int(chooseMix[i])] = muMode[i]
        for i in otherInd:
            meansVec[i] = means[k][i]

        means[k] = meansVec

        covarVec = np.full((nMix, nFeat, nFeat), 0.)
        for i in range(nMode):
            covarVec[int(chooseMix[i])] = sigMode[i]
        for i in otherInd:
            covarVec[i] = (
            1.5 * covars[k][i])  # boost the variance slightly to enable system to adapt to non-spiky data

        covars[k] = covarVec

        if nMode != 0:

            weightVec = np.full(nMix, 0.)
            for i in range(nMode):
                weightVec[int(chooseMix[i])] = modeWght * (wghtMode[i] / sum(wghtMode))
            for i in otherInd:
                weightVec[i] = (1 - modeWght) / len(otherInd)

            weights[k] = weightVec

    return means, covars, weights


def Enforcement(model, X, nCompMax, nMixMax, predZ, numTrySmart, predStateMixCalc, minCoverage, rareState):
    """
    iterate upon the fitted model created with the smart initialization in the hope of better emission dist. estimates
    :param model: the regularized, fitted HMM resulting from a smart initialization of the mixtures
    :param X: the data (n x number Features)
    :param nCompMax: maximum number of states that we are willing to tolerate
    :param nMixMax: maximum number of mixtures that we are willing to tolerate
    :param predZ: the predicted states of the observations according to the model passed as argument
    :param numTrySmart: stores the number of times we've attempted this procedure on "model"
    :param predStateMixCalc: stores the predStateMix calculated when numTrySmart = 0 so we don't redo calculation
    :param minCoverage: minimum coverage achieved by Dominant State Mixtures
    :param rareState: boolean for whether or not we should have a state dedicated to rare State-Mix
    :return: mean, covars and weights of the Gaussian mixtures to be used for a new initialization
    """

    minCompSize = 200  # starting value for finding large state-mixture components
    threshScree = 0.01  # what scree gap will cause us to drop any later eigenvectors
    threshSimultaneity = 0.1
    threshJenShann = 0.3
    n = len(X)
    d = model.n_features

    nMix = model.n_mix

    means = copy.deepcopy(model.means_)
    covars = copy.deepcopy(model.covars_)
    weights = copy.deepcopy(model.weights_)

    # for each obs, finds the predicted (state-mixture) combo. Assumes that the highest prob (state-mixture) will arise
    # from state with highest prob. Calculating rigourously (i.e. over all states) will be quite a bit slower
    if numTrySmart == 0:
        predStateMix = np.full(n, 0)
        for i in range(n):
            try:
                iMix = np.full(nMix, 0.)
                for j in range(nMix):
                    dist = multivariate_normal(mean=means[predZ[i]][j], cov=covars[predZ[i]][j])
                    iMix[j] = dist.pdf(X[i]) * weights[predZ[i]][j]
                predStateMix[i] = (predZ[i] * nMix) + (iMix.argmax())  # i-th obs given label (state * nMix) + mixture
            except:
                print("Cannot calculate likely state-mixture combo for some observations.")
                print([weights[predZ[i]][j], means[predZ[i]][j], covars[predZ[i]][j], X[i]])
    else:
        predStateMix = predStateMixCalc

    print("Stage 1 completed")

    enoughCoverage = False
    freq = np.bincount(predStateMix)  # frequency of each state-mix combo (goes from 0, 1, ..., nComp * nMix - 1)
    while not enoughCoverage:
        ii = np.nonzero(freq > minCompSize)[0]  # give the freqs just of those components with freq above minCompSize
        nMainComp = len(ii)  # number of significant components

        if np.float(sum(freq[ii])) / n > minCoverage:
            enoughCoverage = True

            # for a given threshold minCompSize, we give the indices predicted to lie in each State-Mix combo
            ind = range(n)  # yields [0, 1, ..., nObs - 1]
            predStateMixInd = []
            for i in ii:
                # Recall "predStateMix" is predicted mixture-state combo (from fitted HMM) for each of nObs observations
                isI = predStateMix == i  # vector of Boolean that flags True if predicted state-mix is i.
                subsetInd = [x for x, y in zip(ind, isI) if y]
                predStateMixInd = predStateMixInd + [subsetInd]

            binWidthVec = np.full(nMainComp, 0.)
            for i in range(nMainComp):
                stdDev = np.std(predStateMixInd[i])
                binWidthVec[i] = round(1.75 * stdDev * (freq[ii[i]] ** (-1 / 3)))  # recommended bin Width for variable

            binWidth = np.percentile(binWidthVec, 75)  # set the binWidth so that its wide enough for 75% of state-mix
            # we make the bins just cover the time grid [1, 2,..., n]
            nBins = round(n / binWidth)
            binWidth = np.ceil(n / nBins)

        else:
            minCompSize -= 10

    actualCoverage = np.float(sum(freq[ii])) / n
    stateMixMajor = copy.deepcopy(ii)

    print("Stage 2 completed")

    subsetRareBytes = X
    isRare = [x not in stateMixMajor for x in predStateMix]
    subsetRareBytes = np.array([x for x, y in zip(subsetRareBytes, isRare) if y])
    if len(subsetRareBytes) > 2:
        rareMean = np.mean(subsetRareBytes, axis=0)
        rareCov = np.cov(subsetRareBytes.T)
    else:
        rareMean = np.full((1, d), 0.)
        rareCov = np.full((d, d), 0.)
        for i in range(d):
            rareCov[i, i] = 1

    distModes = np.full((nMainComp, nMainComp), 0.)

    # create bins (each of size binWidth) so that they span the time window [0, nObs]
    upBd = (np.ceil(n / binWidth) * binWidth)
    nBins = int(round(upBd / binWidth))
    bins = [int(x * binWidth) for x in range(nBins + 1)]

    indFreq = np.full((nBins, nMainComp), 0.)

    print("Stage 3 completed")

    # calculate proportions of a particular HMM state/mix component in each of the time bins
    for i in range(nMainComp):
        digitized = np.digitize(predStateMixInd[i], bins) - 1  # we minus one as falling in leftmost bin will then = 0
        yOld = np.bincount(digitized)
        ii = np.nonzero(yOld)[0]

        # y's stop recording frequencies after max obs value
        y = np.array(list(yOld) + [0 for x in range(nBins - max(ii) - 1)])
        # change these to proportions in order to more readily compare various labels
        y = np.array([x / len(predStateMixInd[i]) for x in y])
        iii = range(nBins + 1)

        try:
            indFreq[:, i] = y
        except:
            print(np.vstack((ii, yOld[ii])).T)
            print(np.vstack((iii, y[iii])).T)

    # calculate distance between high density regions based on their relative activity across time
    for i in range(nMainComp):
        for j in range(nMainComp):
            distModes[i, j] = round(spatial.distance.cosine(indFreq[:, i], indFreq[:, j]), 3)

    print("Stage 4 completed")

    numIter = 0
    numNeighbours = 2
    tooManyStates = True
    while tooManyStates:

        if nMainComp > (numNeighbours + 1):
            # create adjacency matrix where adjacent pts are one of the 2 closest pts to node
            # & have dist < threshSimultaneity
            simModes = np.full((nMainComp, nMainComp), 0.)
            for i in range(nMainComp):
                closeInd = np.argpartition(distModes[i, :], (numNeighbours + 1))[0:(numNeighbours + 1)]
                maskInd = distModes[i, closeInd] < threshSimultaneity
                finalInd = [x for x, y in zip(closeInd, maskInd) if y]
                for j in finalInd:
                    simModes[i, j] = 1
                    simModes[j, i] = 1

            diagonal = np.full(nMainComp, 0.)
            for i in range(nMainComp):
                diagonal[i] = np.sum(simModes[i]) ** -0.5
            D = np.diag(diagonal)  # this variable is equal to D^-0.5 in many papers
            Lsym = np.diag(np.full(nMainComp, 1.)) - np.matmul(np.matmul(D, simModes), D)
            vals, vecs = np.linalg.eigh(Lsym)

            eigen_1st_diff = vals[1:nMainComp] - vals[0:(nMainComp - 1)]
            eigen_2nd_diff = eigen_1st_diff[1:(nMainComp - 1)] - eigen_1st_diff[0:(nMainComp - 2)]
            # catches case where gap never > scree gap (which would otherwise return indCut = 0)
            if max(eigen_2nd_diff) > threshScree:
                indCut = np.argmax(eigen_2nd_diff > threshScree) + 2
            else:
                indCut = nMainComp
            preds = cluster.KMeans(n_clusters=indCut, init='k-means++').fit_predict(vecs[:, 1:indCut])

        else:
            indCut = nMainComp
            preds = np.array(range(nMainComp))

        # set maximum number of States to cluster State-Mix components into
        if rareState:
            useComp = nCompMax - 1
        else:
            useComp = nCompMax

        if indCut <= useComp:
            tooManyStates = False
        else:
            numIter += 1

            # if we are struggling to knit states together, increase the number of neighbours
            if numIter >= 6:
                numNeighbours = 3
            if numIter >= 12:
                numNeighbours = 4

            if numIter >= 14:
                tooManyStates = False

                print("Too many iterations required to reduce states. Current distance: %s" % threshSimultaneity)
                # rather cluster using the original distModes Similarity matrix if difficulty with Adjacency matrix
                diagonal = np.full(nMainComp, 0.)
                for i in range(nMainComp):
                    diagonal[i] = np.sum(distModes[i]) ** -0.5
                D = np.diag(diagonal)  # this variable is equal to D^-0.5 in many papers
                Lsym = np.diag(np.full(nMainComp, 1.)) - np.matmul(np.matmul(D, distModes), D)

                print("Decomposing Similarity matrix")
                vals, vecs = np.linalg.eigh(Lsym)
                print("Successful!")

                eigen_1st_diff = vals[1:nMainComp] - vals[0:(nMainComp - 1)]
                eigen_2nd_diff = eigen_1st_diff[1:(nMainComp - 1)] - eigen_1st_diff[0:(nMainComp - 2)]

                # we don't look for the shoulder past the index limEig
                if rareState:
                    limEig = nCompMax - 2
                else:
                    limEig = nCompMax - 1

                # catches case where gap never > scree gap (which would otherwise return indCut = 0)
                if max(eigen_2nd_diff[0:limEig]) > threshScree:
                    indCut = np.argmax(eigen_2nd_diff[0:limEig] > threshScree) + 2
                else:
                    if rareState:
                        indCut = nCompMax - 1
                    else:
                        indCut = nCompMax

                print("Number of states: %s" % indCut)
                preds = cluster.KMeans(n_clusters=indCut, init='k-means++').fit_predict(vecs[:, 1:indCut])
            else:
                threshSimultaneity *= 1.15

    print("Stage 5 completed")

    tooManyMix = True
    while tooManyMix:
        predsDeep = []
        predsDeepCopy = []
        for i in range(indCut):
            subsetNodes = range(nMainComp)
            subsetOriginal = stateMixMajor
            indClass = preds == i

            # indices of major state-mix components assigned to state i
            subsetNodes = [x for x, y in zip(subsetNodes, indClass) if y]
            # org indices of major state-mix components assigned to state i
            subsetOriginal = [x for x, y in zip(subsetOriginal, indClass) if y]

            nSubset = len(subsetNodes)
            JSdist = np.full((nSubset, nSubset), 0.)

            if nSubset >= 4:
                for r in range(nSubset):
                    for s in range(r):
                        state1 = int(np.floor(subsetNodes[r] / nMix))
                        state2 = int(np.floor(subsetNodes[s] / nMix))
                        mix1 = int(subsetNodes[r] % nMix)
                        mix2 = int(subsetNodes[s] % nMix)
                        JSdist[r, s] = jsGaussians(model, state1, state2, mix1, mix2, 1000)
                        JSdist[s, r] = JSdist[r, s]

                simModesI = np.full((nSubset, nSubset), 0.)
                for r in range(nSubset):
                    closeInd = np.argpartition(JSdist[r, :], 3)[0:3]
                    maskInd = JSdist[r, closeInd] < threshJenShann
                    finalInd = [x for x, y in zip(closeInd, maskInd) if y]
                    for s in finalInd:
                        simModesI[r, s] = 1
                        simModesI[s, r] = 1

                diagonalI = np.full(nSubset, 0.)
                for j in range(nSubset):
                    diagonalI[j] = np.sum(simModesI[j]) ** -0.5
                DI = np.diag(diagonalI)  # this variable is equal to D^-0.5 in many papers
                LsymI = np.diag(np.full(nSubset, 1.)) - np.matmul(np.matmul(DI, simModesI), DI)
                valsI, vecsI = np.linalg.eigh(LsymI)

                eigen_1st_diffI = valsI[1:nSubset] - valsI[0:(nSubset - 1)]
                eigen_2nd_diffI = eigen_1st_diffI[1:(nSubset - 1)] - eigen_1st_diffI[0:(nSubset - 2)]
                # catches case where gap never > scree gap (which would otherwise return indCutI = 0)
                if max(eigen_2nd_diffI) > threshScree:
                    indCutI = np.argmax(eigen_2nd_diffI > threshScree) + 2
                else:
                    indCutI = nSubset
                predsI = cluster.KMeans(n_clusters=indCutI, init='k-means++').fit_predict(vecsI[:, 1:indCutI])
            else:
                indCutI = nSubset
                predsI = np.array(range(nSubset))

            predsMixI = []
            predsMixIcopy = []
            for j in range(indCutI):
                subsetNodesI = subsetOriginal
                indClassI = predsI == j
                subsetNodesI = [x for x, y in zip(subsetNodesI, indClassI) if y]

                predsMixI = predsMixI + [subsetNodesI]

                subsetNodesIcopy = subsetNodes
                indClassIcopy = predsI == j
                subsetNodesIcopy = [x for x, y in zip(subsetNodesIcopy, indClassIcopy) if y]
                predsMixIcopy = predsMixIcopy + [subsetNodesIcopy]

            predsDeep = predsDeep + [predsMixI]
            predsDeepCopy = predsDeepCopy + [predsMixIcopy]

        maxComp = max([len(x) for x in predsDeep])
        if maxComp <= nMixMax:
            tooManyMix = False
        else:
            threshJenShann *= 1.05

    print("printing allocation of State-Mixtures to various States & Mixtures")
    print("Maximum Number of Allowed States: %s" % useComp)
    print("Main State-Mix component's indices (indices run over range(nComp * nMix)): %s" % predsDeep)
    print("Main State-Mix component's indices (indices run from over range(nMainComp)): %s" % predsDeepCopy)
    print("Stage 6 completed")

    # calculate the means, covars and weights for the Rare State
    ii = np.nonzero(freq < minCompSize)[0]  # find state-mix components that are rare
    weightRare = freq[ii]  # ... as well as their frequencies
    rareInds = weightRare.argsort()  # order the indices of the rare state-mix components so that freqs are ascending
    topRare = ii[rareInds[::-1]]  # gives rare state-mix components in order of decreasing frequency
    weightRare = weightRare[rareInds[::-1]]  # give corresponding weights for rare state-mix components
    topRare = topRare[0:min(len(topRare), maxComp)]
    if len(topRare) == maxComp:
        weightRare = weightRare[0:maxComp] / sum(weightRare[0:maxComp])
    else:
        if sum(weightRare[0:len(topRare)]) > 0:
            # we weight those state-mix components that exist so that they contain 90% of the rare state's probability
            weightRare = np.array(np.concatenate((0.9 * weightRare[0:len(topRare)] / sum(weightRare[0:len(topRare)]),
                                                  np.full(maxComp - len(topRare), 0.1 / (maxComp - len(topRare))))))
        else:
            weightRare = np.array([0.])

    if rareState:
        if sum(weightRare[0:len(topRare)]) > 0:
            nCompUse = indCut + 1  # we want an extra state to capture the rare classes
        else:
            nCompUse = indCut
    else:
        nCompUse = indCut

    weightNew = np.full((nCompUse, maxComp), -1.)
    meanNew = np.full((nCompUse, maxComp, d), -1.)
    covarNew = np.full((nCompUse, maxComp, d, d), -1.)

    print("Stage 7 completed")
    print([indCut, nCompMax, nCompUse, sum(weightRare[0:len(topRare)])])

    try:
        for i in range(indCut):

            m = 1

            nMixI = len(predsDeep[i])

            m = 2
            unassignedInd = [x + nMixI for x in range(maxComp - nMixI)]

            m = 3
            for j in range(nMixI):
                nCompIJ = len(predsDeep[i][j])  # number of mixtures melded into the j-th mix for state i
                totFreqI = sum(freq[[item for sublist in predsDeep[i] for item in sublist]])  # num obs lying in class i

                m = 5

                ijStates = [int(np.floor(x / nMix)) for x in
                            predsDeep[i][j]]  # state that j-th comp of i-th class lies in
                ijMixs = [int(x % nMix) for x in predsDeep[i][j]]  # mixture that j-th comp of i-th class lies in
                # wght of each of nCompIJ mixtures that lie in j-th component of i-th class
                ijWghts = [freq[x] / sum(freq[predsDeep[i][j]]) for x in predsDeep[i][j]]

                m = 6

                if nMixI == maxComp:
                    weightNew[i][j] = (sum(freq[predsDeep[i][j]]) / totFreqI)
                else:
                    weightNew[i][j] = (sum(freq[predsDeep[i][j]]) / totFreqI) * actualCoverage

                m = 7

                meanNew[i][j] = sum([ijWghts[r] * means[ijStates[r]][ijMixs[r]] for r in range(nCompIJ)])

                m = 8

                covarNew[i][j] = sum([ijWghts[r] * covars[ijStates[r]][ijMixs[r]] for r in range(nCompIJ)])

                m = 9

            m = 10

            for j in unassignedInd:
                weightNew[i][j] = (1 - actualCoverage) / len(unassignedInd)
                meanNew[i][j] = rareMean
                covarNew[i][j] = rareCov
    except BaseException as e:
        print("!!!!!! Error in Enforcement: " + str(e))
        print([i, j, m])
        print([nMix, model.n_mix, model.n_components, [min(predZ), max(predZ)], means.shape, covars.shape])
        print(nCompIJ)
        print(ijWghts)
        print(ijStates)
        print(ijMixs)

    print("Stage 8 completed")

    if rareState:
        try:
            if sum(weightRare[0:len(topRare)]) > 0:
                # set the weights, means and covars for the rare state
                for i in range(maxComp):

                    weightNew[indCut][i] = weightRare[i]

                    if i < len(topRare):
                        stateI = int(np.floor(topRare[i] / nMix))
                        mixI = int(topRare[i] % nMix)

                        meanNew[indCut][i] = means[stateI][mixI]
                        covarNew[indCut][i] = covars[stateI][mixI]
                    else:
                        meanNew[indCut][i] = rareMean
                        covarNew[indCut][i] = rareCov

            sumState = [sum(weightNew[x, :]) for x in range(nCompUse)]
            if (max(sumState) > 1.01) or (min(sumState) < 0.99):
                print("Error: weights not summing to one.")
                print(weightNew)
                print(actualCoverage)
                print(predsDeep)
                print(indCut)
                print(weightRare)
                print(maxComp)
                print(topRare)
        except:
            print("Something went wrong.")

    return nCompUse, maxComp, meanNew, covarNew, weightNew, actualCoverage, threshSimultaneity, threshJenShann, \
           predStateMix


def Aggregation(model, predZ, threshDist, threshScree):
    """
    model calculates the relative simultaneity of states an aggregates states that lie below some threshold
    :param model: the regularized, fitted HMM resulting from a smart initialization of the mixtures
    :param predZ: the predicted states of the observations according to the model passed as argument
    :param threshDist: upper bound of distance between time signatures of states. Below bound, states are merged
    :param threshScree: what value of the 2nd derivative will enable inclusion of corresponding eigenvectors
    :return: weights, means, covariances of the amalgamated states
    """

    nObs = len(predZ)
    nCompOld = model.n_components
    nMix = model.n_mix
    d = model.n_features

    indObs = range(nObs)
    indStateOld = range(nCompOld)

    print("Tying States: Calculating time bins")
    # calculate the appropriate time bins to use
    binWidthVec = np.full(nCompOld, 0.)
    emptyStateVec = np.full(nCompOld, 0.)  # indicator vector equalling 1 when the corresponding state is empty
    for i in range(nCompOld):

        isI = predZ == i
        subsetInd = [x for x, y in zip(indObs, isI) if y]
        nInState = len(subsetInd)

        if nInState > 0:
            stdDev = np.std(subsetInd)
            binWidthVec[i] = round(1.75 * stdDev * (nInState ** (-1 / 3)))  # recommended bin Width for variable
        else:
            emptyStateVec[i] = 1.

    nonEmpty = emptyStateVec < 0.5
    subsetStateInd = [x for x, y in zip(indStateOld, nonEmpty) if y]  # indices of nonempty states

    binWidthVec = binWidthVec[subsetStateInd]  # we restrict the popn of appropriate bin widths to the non-empty states

    binWidth = np.percentile(binWidthVec, 75)  # set the binWidth so that it wide enough for 75% of state-mixture
    # we make the bins just cover the time grid [1, 2,..., n]
    nBins = round(nObs / binWidth)
    binWidth = np.ceil(nObs / nBins)

    # create bins (each of size binWidth) so that they span the time window [0, nObs]
    upBd = (np.ceil(nObs / binWidth) * binWidth)
    nBins = int(round(upBd / binWidth))
    bins = [int(x * binWidth) for x in range(nBins + 1)]

    print(bins)
    print(emptyStateVec)
    print(subsetStateInd)

    nCompNew = len(subsetStateInd)
    indStateNew = range(nCompNew)

    propState = np.full(nCompNew, 0.)  # store the proportion of observations predicted to lie in each state
    indFreq = np.full((len(bins) - 1, nCompNew), 0.)
    distModes = np.full((nCompNew, nCompNew), 0.)

    # calculate proportions of a particular HMM state in each of the time bins
    for i in range(nCompNew):

        isI = predZ == subsetStateInd[i]
        subsetInd = [x for x, y in zip(indObs, isI) if y]
        nInState = len(subsetInd)
        propState[i] = nInState / nObs

        digitized = np.digitize(subsetInd, bins) - 1  # we minus one as falling in leftmost bin will then = 0
        yOld = np.bincount(digitized)
        ii = np.nonzero(yOld)[0]

        # y's stop recording frequencies after max obs value
        y = np.array(list(yOld) + [0 for x in range(len(bins) - max(ii) - 2)])
        # change these to proportions in order to more readily compare various labels
        y = np.array([x / nInState for x in y])

        iii = range(len(bins))

        try:
            indFreq[:, i] = y
        except:
            print(np.vstack((ii, yOld[ii])).T)
            print(np.vstack((iii, y[iii])).T)

    # calculate distance between high density regions based on their relative activity across time
    for i in range(nCompNew):
        for j in range(nCompNew):
            distModes[i, j] = round(spatial.distance.cosine(indFreq[:, i], indFreq[:, j]), 3)

    print("Tying States: Printing simultaneity of states")
    indFreq = np.savetxt(sys.stdout, indFreq, '%5.3f')
    print(indFreq)
    print(distModes)

    # create adjacency matrix where adjacent pts are one of the 2 closest pts to node & have dist<threshSimultaneity
    if nCompNew > 3:
        simModes = np.full((nCompNew, nCompNew), 0.)
        for i in range(nCompNew):
            closeInd = np.argpartition(distModes[i, :], 3)[0:3]
            maskInd = distModes[i, closeInd] < threshDist
            finalInd = [x for x, y in zip(closeInd, maskInd) if y]
            for j in finalInd:
                simModes[i, j] = 1
                simModes[j, i] = 1

        diagonal = np.full(nCompNew, 0.)
        for i in range(nCompNew):
            diagonal[i] = np.sum(simModes[i]) ** -0.5
        D = np.diag(diagonal)  # this variable is equal to D^-0.5 in many papers
        Lsym = np.diag(np.full(nCompNew, 1.)) - np.matmul(np.matmul(D, simModes), D)
        vals, vecs = np.linalg.eigh(Lsym)

        eigen_1st_diff = vals[1:nCompNew] - vals[0:(nCompNew - 1)]
        eigen_2nd_diff = eigen_1st_diff[1:(nCompNew - 1)] - eigen_1st_diff[0:(nCompNew - 2)]
        # catches case where gap never > scree gap (which would otherwise return indCut = 0)
        if max(eigen_2nd_diff) > threshScree:
            indCut = np.argmax(eigen_2nd_diff > threshScree) + 2
        else:
            indCut = nCompNew
        preds = cluster.KMeans(n_clusters=indCut, init='k-means++').fit_predict(vecs[:, 1:indCut])
    else:
        indCut = nCompNew
        simModes = np.full((nCompNew, nCompNew), -1.)
        preds = np.array(range(nCompNew))

    print("Tying States: printing adjacency matrix")
    print(simModes)
    print(preds)

    nCompUse = indCut

    weightNew = np.full((nCompUse, nMix), -1.)
    meanNew = np.full((nCompUse, nMix, d), -1.)
    covarNew = np.full((nCompUse, nMix, d, d), -1.)

    stateAlloc = []
    for i in range(indCut):

        isI = preds == i
        subsetInd = [x for x, y in zip(indStateNew, isI) if y]

        stateAlloc = stateAlloc + [subsetInd]

        if len(subsetInd) == 1:
            weightNew[i] = model.weights_[subsetStateInd[subsetInd[0]]]
            meanNew[i] = model.means_[subsetStateInd[subsetInd[0]]]
            covarNew[i] = model.covars_[subsetStateInd[subsetInd[0]]]
        else:
            indState = []
            indMix = []
            wghtState = []
            totStateProp = sum(propState[subsetInd])
            for j in range(len(subsetInd)):
                indState = indState + [subsetStateInd[subsetInd[j]] for x in range(nMix)]
                indMix = indMix + [x for x in range(nMix)]
                wghtState = wghtState + list((propState[subsetInd[j]] / totStateProp)
                                             * model.weights_[subsetStateInd[subsetInd[j]]])

            indState = np.array([x for _, x in sorted(zip(wghtState, indState), key=lambda pair: pair[0],
                                                      reverse=True)])
            indMix = np.array([x for _, x in sorted(zip(wghtState, indMix), key=lambda pair: pair[0], reverse=True)])
            wghtState = sorted(wghtState, reverse=True)

            indState = indState[0:nMix]
            indMix = indMix[0:nMix]
            wghtState = wghtState[0:nMix]
            totalTopWght = sum(wghtState)
            wghtState = np.array([wghtState[x] / totalTopWght for x in range(nMix)])

            weightNew[i] = wghtState
            for j in range(nMix):
                meanNew[i][j] = model.means_[indState[j]][indMix[j]]
                covarNew[i][j] = model.covars_[indState[j]][indMix[j]]

    return indCut, meanNew, covarNew, weightNew, distModes


def fitGMMHMM(numIterSmart, tailProb, data, dataTrim, structHMM, HDRs, outputPath):
    """
    training module that fits a GMMHMM to data for an entity
    :param numIterSmart: how many successful applications of Enforcement stage are we aiming for
    :param tailProb: how percentage of observations in the extremes of each state are trimmed before fitting GMMHMM
    :param data: user data to be fitted
    :param dataTrim: trimmed version of user data
    :param structHMM: dictionary of parameters related to the model complexity of the GMMHMM
    :param HDRs: dictionary of parameters related to the definition of HDR
    :param outputPath: the directory where all files should be written to
    :return: N/A. All output is written to files.
    """

    nComp = structHMM['nComp']
    nMix = structHMM['nMix']
    extraState = structHMM['extraState']
    extraMix = structHMM['extraMix']
    indContVar = structHMM['indContVar']

    threshVar = HDRs['threshVar']
    threshDist = HDRs['threshDist']
    minObsMode = HDRs['minObsMode']
    minVar = HDRs['minVar']
    minCoverage = HDRs['minCoverage']
    maxCorr = HDRs['maxCorr']
    gmmComp = HDRs['gmmComp']

    bestCoverage = -1.  # remains at -1 if Enforcement plus Iteration can't improve on Initialization + Iteration fit
    # remains at -1 if either no simultaneous states found or simultaneous states found but does not beat best score
    secondEnforceScoreIterate = -1.

    successInitWithIterAndSmartIter = False
    while not successInitWithIterAndSmartIter:

        numTryInitwithIterate = 0
        successFitWithIterate = False
        while not successFitWithIterate:
            nCompUse1 = copy.deepcopy(nComp)
            nMixUse1 = copy.deepcopy(nMix)
            extraStateUse = copy.deepcopy(extraState)
            extraMixUse = copy.deepcopy(extraMix)
            successFit = False
            while not successFit:
                try:
                    jointModel = hmm.GMMHMM(n_components=nCompUse1, n_mix=nMixUse1, covariance_type="full",
                                            init_params="s")

                    # we initialize on data not dataTrim as we don't want trimmed values to be seen as a HDR
                    print("Starting Initialization Procedure")
                    nCompUse1, nMixUse1, means, covars, weights = Initialization(jointModel, data, 12, 50, minObsMode,
                                                                                 threshVar, 0.9, threshDist, indContVar,
                                                                                 extraStateUse, extraMixUse)
                    print("Initialization successful!")
                    jointModel = hmm.GMMHMM(n_components=nCompUse1, n_mix=nMixUse1, covariance_type="full",
                                            init_params="s")
                    jointModel.means_ = copy.deepcopy(means)
                    jointModel.covars_ = copy.deepcopy(covars)
                    jointModel.weights_ = copy.deepcopy(weights)
                    jointModel.transmat_ = matrixCreator(nCompUse1, 0.9)
                    covars = covarRegularizer(jointModel, minVar, maxCorr)
                    jointModel.covars_ = copy.deepcopy(covars)

                    print("Attempting fit on Initialization")
                    jointModel.fit(dataTrim)
                    print("Fit on Initialization succesful")

                    covars = covarRegularizer(jointModel, minVar, maxCorr)
                    jointModel.covars_ = copy.deepcopy(covars)

                    print("Attempting prediction")
                    predZ = jointModel.predict(data)
                    startprob = startprobRegularizer(jointModel, predZ)
                    jointModel.startprob_ = copy.deepcopy(startprob)
                    predZ = jointModel.predict(data)
                    print("Prediction successful")

                    firstScore = jointModel.score(data)
                    print("Initialization Procedure Score: %s" % firstScore)
                    bestScore = copy.deepcopy(firstScore)
                    predZUse = copy.deepcopy(predZ)
                    typeFit = "F"
                    bestModel = copy.deepcopy(jointModel)
                    nCompUse3 = nCompUse1
                    nMixUse3 = nMixUse1
                    successFit = True
                except BaseException as e:
                    print("!!!!!! Initialization Procedure yielded error: " + str(e))
                    if np.abs(nCompUse1 - nMixUse1) > 1:
                        if nCompUse1 > nMixUse1:
                            nCompUse1 -= 1
                        else:
                            nMixUse1 -= 1
                    else:
                        nCompUse1 -= 1
                        nMixUse1 -= 1
                    nCompUse1 = max(nCompUse1, 2)
                    nMixUse1 = max(nMixUse1, 1)

                    # we do away with the need for extra state and mixtures if we need to simplify the model
                    # this prevents us from getting stuck in a loop
                    if nCompUse1 < 4:
                        extraStateUse = 0
                    if nMixUse1 < 5:
                        extraMixUse = 0

                    print([nCompUse1, nMixUse1])
                    print("Initialization Procedure unsuccessful. Reducing the number of components.")

            print("Updating trimmed obs")
            STrimState = trimStates(nCompUse3, dataTrim, predZUse, tailProb)
            print("Successful!")

            try:
                print('Starting Iteration Procedure')
                means, covars, weights = Iteration(jointModel, data, predZUse, 0.9, gmmComp, minObsMode, threshVar * 10,
                                                   indContVar)
                print('Iteration completed')
                jointModel.means_ = copy.deepcopy(means)
                jointModel.covars_ = copy.deepcopy(covars)
                jointModel.weights_ = copy.deepcopy(weights)
                jointModel.transmat_ = matrixCreator(nCompUse1, 0.9)
                covars = covarRegularizer(jointModel, minVar, maxCorr)
                jointModel.covars_ = copy.deepcopy(covars)

                print("Attempting fit on Iteration")
                jointModel.fit(STrimState)
                print("Fit on Iteration successful!")

                covars = covarRegularizer(jointModel, minVar, maxCorr)
                jointModel.covars_ = copy.deepcopy(covars)

                print("Attempting prediction")
                predZ = jointModel.predict(data)
                startprob = startprobRegularizer(jointModel, predZ)
                jointModel.startprob_ = copy.deepcopy(startprob)
                predZ = jointModel.predict(data)
                print("Prediction successful")

                score = jointModel.score(data)
                print("Iteration Procedure Score: %s & Initialization Procedure Score %s" % (score, firstScore))

                if score > firstScore:
                    successFitWithIterate = True
                    initWithIter = True
                    bestScore = copy.deepcopy(score)
                    bestModel = copy.deepcopy(jointModel)
                    predZUse = copy.deepcopy(predZ)
                    typeFit = "E"
                    firstIterate = copy.deepcopy(score)
                    print("Updating trimmed obs")
                    STrimState = trimStates(nCompUse3, dataTrim, predZUse, tailProb)
                    print("Successful!")
                else:
                    numTryInitwithIterate += 1
                    if numTryInitwithIterate > 2:
                        print("Iterations on Initial model are worse. Reverting to initial iteration...")
                        jointModel = copy.deepcopy(bestModel)
                        successFitWithIterate = True
                        initWithIter = False
                        firstIterate = copy.deepcopy(firstScore)
                    else:
                        print("Iteration score worse than Initialization Score. Trying again")
                        print("............... %s ..............." % numTryInitwithIterate)
            except BaseException as e:
                print("!!!!!! Iteration Procedure yielded error: " + str(e))
                firstIterate = -1
                numTryInitwithIterate += 1
                if numTryInitwithIterate > 2:
                    print("Iterations on Initialization Procedure failed. Reverting to Initialization Procedure fit...")
                    jointModel = copy.deepcopy(bestModel)
                    successFitWithIterate = True
                    initWithIter = False
                    firstIterate = copy.deepcopy(firstScore)
                else:
                    print("............... %s ..............." % numTryInitwithIterate)

        smartIterateVec = np.full(numIterSmart, -1.)
        secondIterateVec = np.full(numIterSmart, -1.)

        baselineScore = copy.deepcopy(firstScore)
        everSmartPlusIter = False
        bestEnforce = -999999999999
        minCoverageBase = copy.deepcopy(minCoverage)
        for r in range(numIterSmart):
            print("*************** %s ***************" % r)
            numTrySmart = 0
            # stores the most likely (state-mixture) combo for each obs. We pass this to Enforcement if numTrySmart
            # is greater than zero to avoid calculating it each time
            predStateMix = None
            successSmart = False
            beatEnforce = False
            minCoverageBase = 1 - (0.75 * (1 - minCoverageBase))
            minCoverageUse = copy.deepcopy(minCoverageBase)
            while not successSmart:
                jointModel = copy.deepcopy(bestModel)
                try:
                    # Note that we use nComp and nMix as the state-mixtures post-enforcement will likely have very
                    # litte relation to the number of States/Mixtures of the model being fed into the Enforcement stage
                    print("Starting Enforcement Procedure")
                    nCompUse2, nMixUse2, means, covars, weights, coverage, sim, JS, predStateMix = Enforcement(
                        jointModel, data, nComp, nMix, predZUse, numTrySmart, predStateMix, minCoverageUse, True)
                    print("Enforcement completed with minimum distance: %s & coverage threshold: %s" % (sim,
                                                                                                        minCoverageUse))

                    # if distance between "simultaneous states" too large, we reduce number of dominant state-mixtures
                    # before passing to Enforcement stage via the reduction of the minimum Coverage required
                    if sim > 0.2:
                        print("Minimum distance for adjacency between State-Mixtures not great. Trying again.")
                        print("Changing coverage. Old coverage: %s" % minCoverageUse)
                        minCoverageUse *= 0.8
                        print("New coverage: %s" % minCoverageUse)
                        nCompUse2, nMixUse2, means, covars, weights, coverage, sim, JS, predStateMix = Enforcement(
                            jointModel, data, nComp, nMix, predZUse, numTrySmart + 1,
                            predStateMix, minCoverageUse, True)
                        print("Enforcement completed with minimum distance: %s & coverage threshold: %s" % (
                            sim, minCoverageUse))

                    jointModel = hmm.GMMHMM(n_components=nCompUse2, n_mix=nMixUse2, covariance_type="full",
                                            init_params="s")
                    jointModel.means_ = copy.deepcopy(means)
                    jointModel.covars_ = copy.deepcopy(covars)
                    jointModel.weights_ = copy.deepcopy(weights)
                    jointModel.transmat_ = matrixCreator(nCompUse2, 0.9)
                    covars = covarRegularizer(jointModel, minVar, maxCorr)
                    jointModel.covars_ = copy.deepcopy(covars)

                    print("Attempting fit on Enforcement")
                    jointModel.fit(STrimState)
                    print("Fit on Enforcement successful")

                    covars = covarRegularizer(jointModel, minVar, maxCorr)
                    jointModel.covars_ = copy.deepcopy(covars)

                    print("Attempting prediction")
                    predZ = jointModel.predict(data)
                    startprob = startprobRegularizer(jointModel, predZ)
                    jointModel.startprob_ = copy.deepcopy(startprob)
                    predZ = jointModel.predict(data)
                    print("Prediction successful")

                    print("Scoring Model")
                    score = jointModel.score(data)
                    smartIterModel = copy.deepcopy(jointModel)
                    successInitWithIterAndSmartIter = True
                    print("Enforcement Procedure Score: %s & Baseline Procedure score %s" % (score, baselineScore))

                    lastEnforce = copy.deepcopy(score)

                    if score > bestEnforce:
                        bestEnforce = score
                        beatEnforce = True
                        smartIterate = copy.deepcopy(score)
                    else:
                        beatEnforce = False

                    if score > baselineScore:
                        successSmart = True
                        successInitWithIterAndSmartIter = True
                        smartIterate = copy.deepcopy(score)
                        baselineScore = copy.deepcopy(smartIterate)

                        bestModel = copy.deepcopy(jointModel)
                        bestScore = copy.deepcopy(smartIterate)
                        bestCoverage = copy.deepcopy(minCoverageUse)
                        bestSim = copy.deepcopy(sim)
                        bestJS = copy.deepcopy(JS)
                        predZUse = copy.deepcopy(predZ)
                        nCompUse3 = nCompUse2
                        nMixUse3 = nMixUse2
                        print("Updating trimmed obs")
                        STrimState = trimStates(nCompUse3, dataTrim, predZUse, tailProb)
                        print("Successful!")
                        if not everSmartPlusIter:
                            if initWithIter:
                                typeFit = "B"
                                print("Score on Enforcement after (Initialization + Iteration): %s" % smartIterate)
                            else:
                                typeFit = "D"
                                print("Score on Enforcement after just Initialization Procedure: %s" % smartIterate)
                    else:
                        numTrySmart += 1
                        print("Enforcement Score worse than baseline score.")
                        if numTrySmart > 2:
                            successInitWithIterAndSmartIter = True
                            smartIterate = copy.deepcopy(bestScore)

                            # we don't want to revert to previous model in HMM if Enforcement at least beats previous
                            # Enforcement best as we will rather perform iteration and see if that beats baseline score
                            if not beatEnforce:
                                jointModel = copy.deepcopy(bestModel)  # revert back to best fitting model

                            if everSmartPlusIter:
                                print("Best model uses either Enforcement alone or (Weak Enforcement + Iteration)")
                            else:
                                if successSmart:
                                    print("Best model is Enforcement w\o Iteration after multiple trials")
                                else:
                                    print("Best model is (Initialization, Iteration) after multiple Enforcement trials")
                                    print(jointModel.transmat_)
                                    bestSim = -1.
                                    bestJS = -1.

                            successSmart = True

                        else:
                            print("Changing coverage. Old coverage: %s" % minCoverageUse)
                            minCoverageUse = 1 - (0.75 * (1 - minCoverageUse))
                            print("Changed minimum coverage to: %s and re-attempting Enforcement" % minCoverageUse)

                            if not beatEnforce:  # case where we don't run the Iteration procedure
                                print("::::::::::::::: %s :::::::::::::::" % numTrySmart)
                            else:
                                print("Enforcement bested its best, but not baseline. We pass it to Iteration")

                    if beatEnforce:
                        numTryIter = 0
                        successIter = False
                        while not successIter:
                            try:
                                means, covars, weights = Iteration(jointModel, data, predZUse, 0.9, gmmComp, minObsMode,
                                                                   threshVar * 10, indContVar)
                                jointModel.means_ = copy.deepcopy(means)
                                jointModel.covars_ = copy.deepcopy(covars)
                                jointModel.weights_ = copy.deepcopy(weights)
                                jointModel.transmat_ = matrixCreator(nCompUse2, 0.9)
                                covars = covarRegularizer(jointModel, minVar, maxCorr)
                                jointModel.covars_ = copy.deepcopy(covars)

                                print("Starting Iteration Procedure after Enforcement Stage")
                                jointModel.fit(STrimState)
                                print("Fit on Iteration successful!")

                                covars = covarRegularizer(jointModel, minVar, maxCorr)
                                jointModel.covars_ = copy.deepcopy(covars)

                                print("Attempting prediction")
                                predZ = jointModel.predict(data)
                                startprob = startprobRegularizer(jointModel, predZ)
                                jointModel.startprob_ = copy.deepcopy(startprob)
                                predZ = jointModel.predict(data)
                                print("Prediction successful")

                                score = jointModel.score(data)
                                print("Enforcement + Iteration Score: %s & Baseline score %s" % (score, baselineScore))

                                # 2nd condition ensures that we update model to enforcement & iteration if the score
                                # is at least better than the first stage
                                if (score > bestScore) or ((not everSmartPlusIter) and (score > baselineScore)):
                                    secondIterate = copy.deepcopy(score)
                                    bestModel = copy.deepcopy(jointModel)
                                    predZUse = copy.deepcopy(predZ)

                                    bestScore = copy.deepcopy(secondIterate)
                                    bestCoverage = copy.deepcopy(minCoverageUse)
                                    bestSim = copy.deepcopy(sim)
                                    bestJS = copy.deepcopy(JS)

                                    # Would be unnecessary as Iteration does not change nComp/nMix... except that it
                                    # covers the case where best model is updated to have same State-Mixtures as weak
                                    # Enforcement (in cases where Weak Enforcement + Iteration > baselineScore)
                                    nCompUse3 = nCompUse2
                                    nMixUse3 = nMixUse2

                                    print("Updating trimmed obs")
                                    STrimState = trimStates(nCompUse3, dataTrim, predZUse, tailProb)
                                    print("Successful!")

                                    everSmartPlusIter = True
                                    # we say that enforcement is successful if (Enforcement + Iteration) beats baseline
                                    successSmart = True
                                    if initWithIter:
                                        typeFit = "A"
                                        print("Score on Fully Successful Iteration: %s" % secondIterate)
                                    else:
                                        typeFit = "C"
                                        print("Score on Smart & Iteration after failed Iteration: %s" % secondIterate)

                                    successIter = True
                                else:
                                    numTryIter += 1
                                    if numTryIter > 2:
                                        successIter = True
                                        jointModel = copy.deepcopy(bestModel)
                                        secondIterate = copy.deepcopy(bestScore)
                                    else:
                                        print("Iteration Fit not used.")
                                        print("............... %s ..............." % numTryIter)

                                if successIter:
                                    secondIterateVec[r] = secondIterate

                            except BaseException as e:
                                print("!!!!!! Iteration on Enforcement Procedure failed with error: " + str(e))
                                print("Reverting to Best Available Model...")
                                jointModel = copy.deepcopy(bestModel)  # smart Iteration beat previous model
                                secondIterate = -1.
                                numTryIter += 1
                                if numTryIter > 2:
                                    successIter = True
                                else:
                                    print("............... %s ..............." % numTryIter)

                    if successSmart:
                        smartIterateVec[r] = smartIterate

                except BaseException as e:
                    print("!!!!!! Enforcement Procedure yielded error " + str(e))
                    smartIterate = -1
                    numTrySmart += 1
                    print("Changing coverage. Old coverage: %s" % minCoverageUse)
                    minCoverageUse *= 0.9
                    print("New Coverage: %s" % minCoverageUse)
                    print(str(e))
                    if numTrySmart > 2:
                        successSmart = True
                        successInitWithIterAndSmartIter = True
                        jointModel = copy.deepcopy(bestModel)
                        smartIterate = copy.deepcopy(bestScore)
                        secondIterate = copy.deepcopy(bestScore)
                        smartIterateVec[r] = smartIterate
                        secondIterateVec[r] = secondIterate
                    else:
                        print("............... %s ..............." % numTrySmart)

    print("-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-")
    try:
        nCompUse2, means, covars, weights, distModes = Aggregation(jointModel, predZUse, 0.075, 0.01)
        print([nCompUse2, nCompUse3, bestScore, nMixUse3])
        if nCompUse2 < nCompUse3:
            jointModel = hmm.GMMHMM(n_components=nCompUse2, n_mix=nMixUse3, covariance_type="full",
                                    init_params="s")
            jointModel.means_ = copy.deepcopy(means)
            jointModel.covars_ = copy.deepcopy(covars)
            jointModel.weights_ = copy.deepcopy(weights)
            jointModel.transmat_ = matrixCreator(nCompUse2, 0.9)
            covars = covarRegularizer(jointModel, minVar, maxCorr)
            jointModel.covars_ = copy.deepcopy(covars)

            numTry = 0
            successTie = False
            while (numTry < 2) and (not successTie):
                try:
                    print("Tying States: Attempting fit")
                    jointModel.fit(STrimState)
                    print("Tying States: Fit successful")
                    successTie = True
                except BaseException as e:
                    numTry += 1
                    print([jointModel.n_components, jointModel.n_mix])
                    print("!!!!!! Fit when making aggregating simultaneous states failed with error: " + str(e))

            covars = covarRegularizer(jointModel, minVar, maxCorr)
            jointModel.covars_ = copy.deepcopy(covars)

            print("Attempting prediction")
            predZ = jointModel.predict(data)
            startprob = startprobRegularizer(jointModel, predZ)
            jointModel.startprob_ = copy.deepcopy(startprob)
            predZ = jointModel.predict(data)
            print("Prediction successful")

            secondEnforceScore = jointModel.score(data)
            bestScore = copy.deepcopy(secondEnforceScore)
            predZUse = copy.deepcopy(predZ)
            bestModel = copy.deepcopy(jointModel)

            print("Enforcing Simultaniety changed score from %s to %s" % (baselineScore, secondEnforceScore))

            nCompUse3 = nCompUse2
            print("Updating trimmed obs")
            STrimState = trimStates(nCompUse3, dataTrim, predZUse, tailProb)
            print("Succesful!")

            numTryIter = 0
            successIter = False
            while not successIter:
                try:
                    means, covars, weights = Iteration(jointModel, data, predZUse, 0.9, gmmComp, minObsMode,
                                                       threshVar * 10, indContVar)
                    jointModel.means_ = copy.deepcopy(means)
                    jointModel.covars_ = copy.deepcopy(covars)
                    jointModel.weights_ = copy.deepcopy(weights)
                    jointModel.transmat_ = matrixCreator(nCompUse2, 0.9)
                    covars = covarRegularizer(jointModel, minVar, maxCorr)
                    jointModel.covars_ = copy.deepcopy(covars)

                    print("Trying to fit Iteration on Enforcement Stage")
                    jointModel.fit(STrimState)
                    print("Fit successful!")

                    covars = covarRegularizer(jointModel, minVar, maxCorr)
                    jointModel.covars_ = copy.deepcopy(covars)

                    print("Attempting prediction")
                    predZ = jointModel.predict(data)
                    startprob = startprobRegularizer(jointModel, predZ)
                    jointModel.startprob_ = copy.deepcopy(startprob)
                    predZ = jointModel.predict(data)
                    print("Prediction successful")

                    score = jointModel.score(data)

                    if score > secondEnforceScore:
                        bestModel = copy.deepcopy(jointModel)
                        secondEnforceScoreIterate = copy.deepcopy(score)
                        bestScore = copy.deepcopy(secondEnforceScoreIterate)
                        predZUse = copy.deepcopy(predZ)

                        successIter = True
                    else:
                        numTryIter += 1
                        if numTryIter > 2:
                            successIter = True
                            jointModel = copy.deepcopy(bestModel)

                except BaseException as e:
                    print("Iteration on Tying of Simultaneuous States failed with error: " + str(e))
                    print("Reverting to smart iteration...")
                    jointModel = copy.deepcopy(bestModel)  # smart Iteration beat previous model
                    numTryIter += 1
                    if numTryIter > 2:
                        successIter = True

        else:
            print("No simultaneous states found!!")
            secondEnforceScore = -1
            secondEnforceScoreIterate = -1
    except BaseException as e:
        print("Attempting to find simultaneous states failed with error: " + str(e))
        jointModel = copy.deepcopy(bestModel)
        secondEnforceScore = -1
        secondEnforceScoreIterate = -1
        distModes = np.full((1, 1), -1.)

    # we also store for each observation, the probabilities of lying in the different states for use by plotting fns
    predProb = jointModel.predict_proba(data)

    fitStats = dict([('bestScore', bestScore), ('firstScore', firstScore), ('firstIterate', firstIterate),
                     ('smartIterateVec', smartIterateVec), ('secondIterateVec', secondIterateVec),
                     ('secondEnforceScore', secondEnforceScore), ('typeFit', typeFit),
                     ('secondEnforceScoreIterate', secondEnforceScoreIterate), ('bestCoverage', bestCoverage),
                     ('bestSim', bestSim), ('bestJS', bestJS)])

    outputPath += "fileComm/"
    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)
    outputPath += "training/"
    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)
    modelPath = outputPath + "model.sav"
    distModesPath = outputPath + "distModes.csv"
    modelStructPath = outputPath + "modelStateMix.csv"
    fitStatsPath = outputPath + "fitStats.csv"
    predZpath = outputPath + "predZ.csv"
    predProbPath = outputPath + "predProb.csv"

    pickle.dump(jointModel, open(modelPath, 'wb'))
    np.savetxt(distModesPath, distModes, delimiter=",")
    np.savetxt(predZpath, predZUse, delimiter=",")
    np.savetxt(predProbPath, predProb, delimiter=",")

    with open(modelStructPath, 'w') as resultFile:
        wr = csv.writer(resultFile, dialect='excel')
        wr.writerow([nCompUse1, nCompUse3, nMixUse1, nMixUse3])

    with open(fitStatsPath, 'w') as f:
        w = csv.DictWriter(f, fitStats.keys())
        w.writeheader()
        w.writerow(fitStats)
