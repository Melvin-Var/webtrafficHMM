from scipy.stats import norm
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl


def gaussianMixture(x, means, variances, weights, iFeat):
    """
    Gaussian Mixture pdf
    :param x: where to evaluate Gaussian mixture
    :param means: length nMix vector of means for the Gaussian components
    :param variances: corresponding vector of variances of the Gaussian components
    :param weights: corresponding vector of weights for the Gaussian components
    :param iFeat: index of feature for which we wish to find the Gaussian component
    :return: Gaussian mixture's pdf value at x
    """

    nMix = len(means)
    dens = 0
    for i in range(nMix):
        dens += weights[i] * norm(means[i][iFeat], np.sqrt(variances[i][iFeat][iFeat])).pdf(x)

    return dens


def multivarEmissionPlotter(resol, nGrid, model, X, predZ, featStr, workDir, ypos):
    """
    plots the emission probabilites per state per feature & scatter plots of data partitioned by predicted state
    :param resol: used to determine buffer above/below min/max value to go for x/y axis of both plots
    :param nGrid: number of grid lines to use for x/y axis
    :param model: final, fitted HMM model
    :param X: the data (n x number Features)
    :param predZ: predicted states for data using the HMM model
    :param featStr: list of strings used to give the titles for each subplot
    :param ypos: (float) where to position the titles of emission plots
    :return: nothing!
    """

    nComp = len(model.startprob_)
    n = len(X)
    nFeat = len(X[0])
    mpl.rcParams['ytick.labelsize'] = 4
    mpl.rcParams['xtick.labelsize'] = 4
    # mpl.rcParams['axes.titlepad'] = -3

    grid = np.full((nFeat, nGrid), 0.)  # will store the grids for each of the nFeat features
    for i in range(nFeat):
        var = [X[x][i] for x in range(n)]
        grid[i] = np.linspace(np.floor(min(var) / resol) * resol, np.ceil(max(var) / resol) * resol, nGrid)

    fig, axes = plt.subplots(nrows=1, ncols=nFeat)
    plt.subplots_adjust(wspace=0.4)
    for col, big_ax in enumerate(axes, start=1):
        big_ax.set_aspect(0.5625)
        big_ax.set_title(featStr[col - 1], fontsize=8, y=ypos)

        # Turn off axis lines and ticks of the big subplot
        # obs alpha is 0 in RGBA string!
        big_ax.tick_params(labelcolor=(1., 1., 1., 0.0), top='off', bottom='off', left='off', right='off')
        # removes the white frame
        big_ax._frameon = False

    cmap = mpl.cm.get_cmap('Paired')
    for j in range(nFeat):
        lens = np.full(nComp, None)

        for i in range(nComp):

            # find the subset of bytes that are predicted to belong to state i
            subsetBytes = [x[j] for x in X]
            isI = predZ == i  # vector of Boolean that flags True if predicted state is i
            subsetBytes = [x for x, y in zip(subsetBytes, isI) if y]
            lens[i] = round(len(subsetBytes) / len(X), 3)

            yVals = [gaussianMixture(x, model.means_[i], model.covars_[i], model.weights_[i], j) for x in grid[j]]

            ax = fig.add_subplot(nComp, nFeat, ((i * nFeat) + j + 1))
            ax.hist(subsetBytes, normed=True, bins=80)
            ax.set_ylabel('$b_{%s}(x)$' % i, fontsize=4)
            # ax.set_yticklabels(labels=['', ''], fontsize=6)
            if j == 0:
                ax.annotate('Pr: %s' % lens[i], xy=(0.75, 0.7), xycoords='axes fraction', size=4)
            if i != nComp - 1:
                ax.tick_params(
                    axis='x',  # changes apply to the x-axis
                    which='both',  # both major and minor ticks are affected
                    bottom='off',  # ticks along the bottom edge are off
                    top='off',  # ticks along the top edge are off
                    labelbottom='off')
            ax.plot(grid[j], yVals, linewidth=0.8)  # , c=rgba)

    plt.savefig(workDir, bbox_inches='tight', figsize=(1024 / 200, 2800 / 200), dpi=800)
    plt.close()
    # plt.show()


def HMMplotting(X, model, predZ, plotPar, scoreVec, ssiScore, predClass, outputPath, featStr, fileStr, score, typeFit):
    """
    yields a bunch of plots of model output
    :param X: data to be plotted (n x number Features) with model augmentations
    :param model: final GMMHMM model used for plotting
    :param predZ: predicted states for the observations
    :param plotPar: parameters required for plotting Emission distributions
    :param scoreVec: vector of GMMHMM scores
    :param ssiScore: vector of SSI scores
    :param predClass: predicted clas of the observations
    :param outputPath: directory where to save the evaluation output
    :param featStr: list of strings containing feature names
    :param fileStr: common string shared by all the plot outputs
    :param score: contains the model score
    :param typeFit: (A = Iterate on Smart, B = Smart w/o Iterate, C = Iterate w/o Smart, D = Basic Fit)
    """

    cutPr = plotPar["cutPr"]
    resol = plotPar['resol']
    nGrid = plotPar['nGrid']
    ypos = plotPar['ypos']

    nFeat = len(X[0])
    n = len(X)
    nComp = model.n_components
    nMix = model.n_mix

    filenamePP = outputPath + "PP_" + fileStr + ".png"
    filenameScore = outputPath + "Detections" + fileStr
    workDirEm = outputPath + "Emission_" + fileStr + ".png"

    # start the PP plots
    scoreSort = np.sort(scoreVec)
    empCDF = np.linspace((1 / len(X)), 1, len(X))

    mpl.rcParams['ytick.labelsize'] = 4
    mpl.rcParams['xtick.labelsize'] = 4

    plt.figure()
    plt.plot(empCDF, empCDF)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.scatter(empCDF, scoreSort, s=0.2)
    plt.xlabel("Empirical CDF")
    plt.ylabel("Theoretical CDF")
    plt.title("PP-plot")
    plt.savefig(filenamePP, bbox_inches='tight', figsize=(1022 / 200, 1873 / 200), dpi=800)
    plt.close()

    # created augmented vectors to use in scatter plots
    outlierBytes = [x for x in X]
    xOut = range(len(X))
    # isOut = scoreVec > cutPr
    isOut = scoreVec < cutPr
    outlierBytes = np.array([x for x, y in zip(outlierBytes, isOut) if y])
    xOut = [x for x, y in zip(xOut, isOut) if y]
    # scoreOut = [x for x, y in zip(scoreVec, isOut) if y]
    scoreOut = [np.log(x + 1e-16) for x, y in zip(scoreVec, isOut) if y]

    outlierBytesStd = [x for x in X]
    xOutStd = range(len(X))
    xOutMidStd = range(len(X))
    isOutStd = ssiScore > 0
    isOutMidStd = ssiScore > -2.197
    outlierBytesStd = np.array([x for x, y in zip(outlierBytesStd, isOutStd) if y])
    xOutStd = [x for x, y in zip(xOutStd, isOutStd) if y]  # list of indices for SSI outliers
    xOutMidStd = [x for x, y in zip(xOutMidStd, isOutMidStd) if y]  # list of indices for (mid) SSI outliers

    sVec = np.full(len(scoreVec), 0.)
    for i in range(len(scoreVec)):
        if scoreVec[i] < cutPr:
            # if scoreVec[i] > cutPr:
            sVec[i] = 2
        else:
            sVec[i] = 0.1

        if ssiScore[i] > 0:
            sVec[i] = 10
        else:
            if ssiScore[i] > -2.197:
                sVec[i] = 6

    print(xOutStd)
    print(outlierBytesStd)
    print([len(xOut), len(xOutMidStd), len(xOutStd)])

    if len(scoreOut) > 0:
        # thresh = min(scoreOut)
        thresh = max(scoreOut)
        # we don't want the colour gradient to be wasted discriminating between relative outlier likelihoods
        # colourVec = [-1 * min(x, thresh) for x in scoreVec]
        # print([colourVec[x] for x in xOut])
        colourVec = [max(np.log(x + 1e-16), thresh) for x in scoreVec]
    else:
        # colourVec = [-1 * x for x in scoreVec]
        colourVec = [np.log(x) for x in scoreVec]

    # create the scatter plots
    fig, axes = plt.subplots(nrows=(2 * nFeat), ncols=1)
    for i in range(nFeat):
        axes[i].scatter(range(n), [x[i] for x in X], s=sVec, c=colourVec, alpha=0.8, cmap="inferno")
        axes[i].set_ylabel(featStr[i], fontsize=8)
        if len(xOut) > 0:
            axes[i].scatter(xOut, [x[i] for x in outlierBytes], s=[sVec[x] for x in xOut],
                            c=[colourVec[x] for x in xOut], cmap="inferno")
        if len(xOutStd) > 0:
            axes[i].scatter(xOutStd, [x[i] for x in outlierBytesStd], s=[sVec[x] for x in xOutStd],
                            c=[colourVec[x] for x in xOutStd], cmap="inferno")

        axes[nFeat + i].scatter(range(n), [x[i] for x in X], s=sVec, c=(predClass / nComp), alpha=0.5, cmap="tab20")
        axes[nFeat + i].set_ylabel(featStr[i], fontsize=8)
        if len(xOut) > 0:
            axes[nFeat + i].scatter(xOut, [x[i] for x in outlierBytes], s=[sVec[x] for x in xOut],
                                    c=[predClass[x] / nComp for x in xOut], cmap="tab20")
        if len(xOutStd) > 0:
            axes[nFeat + i].scatter(xOutStd, [x[i] for x in outlierBytesStd], s=[sVec[x] for x in xOutStd],
                                    c=[predClass[x] / nComp for x in xOutStd], cmap="tab20")

        if i < (nFeat - 1):
            axes[i].tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
            axes[nFeat + i].tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
        else:
            axes[i].tick_params(axis='x', which='both', bottom='off', top='off', labelbottom='off')
            axes[nFeat + i].set_xlabel("index", fontsize=8)

    parNum = (nComp * nComp) + (3 * nComp * nMix) - nComp + 1
    filenameScore = filenameScore + "_" + "[" + str(nComp) + "," + str(nMix) + "," + str(parNum) + "," + str(n) \
                    + "," + str(round(score, 1)) + typeFit + "]" + "_" + str(len(xOut)) + "_" \
                    + str(len(xOutMidStd)) + "_" + str(len(xOutStd)) + ".png"
    plt.savefig(filenameScore, bbox_inches='tight', figsize=(1022 / 200, 1873 / 200), dpi=800)
    plt.close()

    multivarEmissionPlotter(resol, nGrid, model, X, predZ, featStr, workDirEm, ypos)
