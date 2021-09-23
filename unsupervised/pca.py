import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AffinityPropagation
import sklearn.cluster as cluster
import csv
import os, sys
import matplotlib.pyplot as plt

train_path = sys.argv[1]
output_path = sys.argv[2]

def plot_scatter(feat, label, savefig=None):
    """ Plot Scatter Image.
    Args:
      feat: the (x, y) coordinate of clustering result, shape: (9000, 2)
      label: ground truth label of image (0/1), shape: (9000,)
    Returns:
      None
    """
    X = feat[:, 0]
    Y = feat[:, 1]
    plt.scatter(X, Y, c = label)
    plt.legend(loc='best')
    if savefig is not None:
        plt.savefig(savefig)
    plt.show()
    return

def cal_acc(gt, pred):
    """ Computes categorization accuracy of our task.
    Args:
      gt: Ground truth labels (9000, )
      pred: Predicted labels (9000, )
    Returns:
      acc: Accuracy (0~1 scalar)
    """
    # Calculate Correct predictions
    correct = np.sum(gt == pred)
    acc = correct / gt.shape[0]
    # 因為是 binary unsupervised clustering，因此取 max(acc, 1-acc)
    return max(acc, 1-acc)

def save_prediction(pred, out_csv='prediction.csv'):
    with open(out_csv, 'w') as f:
        f.write('id,label\n')
        for i, p in enumerate(pred):
            f.write(f'{i},{p}\n')
    print(f'Save prediction to {out_csv}.')

def predict(data):
    batch = data.shape[0]
    latents = np.reshape(data, (batch, -1))
    
    pca = PCA(n_components=400, whiten=True, random_state=0)
    p = pca.fit_transform(latents)
    
    # # Second Dimesnion Reduction
    k = KMeans(n_clusters=2, random_state=0).fit(p)

    pred = [int(i) for i in k.labels_]
    pred = np.array(pred)
    return pred

def invert(pred):
    return np.abs(1-pred)

trainX = np.load(train_path)
pred = predict(trainX)
#save_prediction(pred, output_path)
save_prediction(invert(pred), output_path)
