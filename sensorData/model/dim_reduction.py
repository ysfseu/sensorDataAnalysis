"""Reduce the dimentsion to 2 or 3 to visualize the data
"""
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.lda import LDA
from matplotlib import pyplot as plt
import numpy as np
import sys, traceback
def dim_reduction_PCA(X,n_dim):
    """ Reduce the dimension by PCA.

    :param X: matrix data (n*k), n is the number of samples. k is the dimension of each sample
    :param n_dim: number of dimension we desired to reduce to.
    :return reduced_X:matrix data(n*n_dim)
    """

    try:
        reduced_X = sklearnPCA(n_components=n_dim).fit_transform(X)
    except:
        print ("Dimension Error")
        reduced_X = []
    finally:
        return reduced_X

def dim_reduction_LDA(X,Y,n_dim):
    """ Reduce the dimension by PCA.

    :param X: matrix data (n*k), n is the number of samples. k is the dimension of each sample
    :param n_dim: number of dimension we desired to reduce to.
    :param Y: reference or labels
    :return reduced_X:matrix data(n*n_dim)
    """
    try:
        reduced_X = LDA(n_components=n_dim).fit_transform(X,Y)
    except:
        print "dimension error"
        reduced_X = X
    finally:
        return np.array(reduced_X)

def plot_data(reduced_X,Y,title,mirror=1):
    """ visualize low dimension data.

    :param reduced_X: matrix data (n*k), n is the number of samples. k is the dimension of each sample
    :param Y: reference or labels
    :param: title of the figure
    """
    try:
        n_dim = reduced_X.shape[1]
        plt.figure()
        ax = plt.subplot(111)
        if n_dim ==1:
            for label,marker,color in zip(
                range(len(set(Y))),('^', 's', 'o'),('blue', 'red', 'green')):
                plt.scatter(x=reduced_X[Y == label]*mirror, y=np.zeros_like(reduced_X[Y == label]),
                    marker=marker,
                    color=color,
                    alpha=0.5
                    )
        else:
            for label,marker,color in zip(range(len(set(Y))),('^', 's','+','*'),('blue', 'red','green','orange')):

                plt.scatter(x=reduced_X[:,0][Y == label]*mirror,
                        y=reduced_X[:,1][Y == label],
                        marker=marker,
                        color=color,
                        alpha=0.5
                    )

        plt.xlabel('Dim1')
        plt.ylabel('Dim2')

        leg = plt.legend(loc='upper right', fancybox=True)
        #leg.get_frame().set_alpha(0.5)
        plt.title(title)

        # hide axis ticks
        plt.tick_params(axis="both", which="both", bottom="off", top="off",
            labelbottom="on", left="off", right="off", labelleft="on")

        # remove axis spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)

        plt.grid()
        plt.tight_layout
        plt.show()
        path = "./resultats/"
        #plt.savefig(path+title)
    except:
        print("erro")
        traceback.print_exc(file=sys.stdout)
        return
