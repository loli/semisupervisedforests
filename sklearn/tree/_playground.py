#!/usr/bin/python

import numpy as np
from sklearn.ensemble import UnSupervisedRandomForestClassifier
import matplotlib.pyplot as plt

def main():
    n_samples, n_features = 100, 2
    np.random.seed(42)
    
    yl = []
    Xl = []
    center = np.asarray([0,0])
    center_step = np.asarray([5,5])
    for i in range(10):
        cov = np.diagflat([1] * n_features)
        _X = np.random.multivariate_normal(center, cov, n_samples)
        _y = [i] * n_samples
        Xl.append(_X)
        yl.append(_y)
        center += center_step
        
    center = np.asarray([0,50])
    center_step = np.asarray([5,-5])
    for i in range(10):
        cov = np.diagflat([1] * n_features)
        _X = np.random.multivariate_normal(center, cov, n_samples)
        _y = [i] * n_samples
        Xl.append(_X)
        yl.append(_y)
        center += center_step        
        
    y = np.concatenate(yl)
    X = np.vstack(Xl)
    
    uc = UnSupervisedRandomForestClassifier(n_estimators=1, max_depth=5, max_features=None, bootstrap=False) #, min_samples_leaf=60)
    uc.fit(X, y)
    #yr = uc.predict(X)
    
    t0 = uc.estimators_[0].tree_
    
    print 'features:', t0.feature
    print 'impurity:', t0.impurity
    print 'threshold', t0.threshold

    #print '{}/{}'.format(np.count_nonzero(yr == y), n_samples * 2)
    
    #####
    # PLOT
    #####
    colours = ['r', 'g', 'b']
    for i, _X in enumerate(Xl):
        c = colours[i % len(colours)]
        plt.scatter(_X[:,0], _X[:,1], c=c)
    
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    lines, _ = tree_lines(t0, [xmin, ymin], [xmax, ymax])
    lines_x = np.asarray([(x, ymin[1], ymax[1]) for fid, x, ymin, ymax in lines if 0 == fid])
    lines_y = np.asarray([(y, xmin[0], xmax[0]) for fid, y, xmin, xmax in lines if 1 == fid])
    
    if not 0 == lines_x.size: plt.vlines(lines_x[:,0], lines_x[:,1], lines_x[:,2], linestyles=u'dashed')
    if not 0 == lines_y.size:plt.hlines(lines_y[:,0], lines_y[:,1], lines_y[:,2], linestyles=u'dashed')
    
    plt.show()
    
def tree_lines(tree, min, max, pos = 0, lines = []):
    # init
    if not type(min) is list:
        min = [min] * tree.n_features
    if not type(max) is list:
        max = [max] * tree.n_features
        
    fid = tree.feature[pos]
    
    # reached leaf
    if -2 == fid:
        return lines, pos
    
    # normal split node: add line
    thr = tree.threshold[pos]
    lines.append([fid, thr, min[:], max[:]])
    
    # ascend to left node
    max_left = max[:]
    max_left[fid] = thr
    lines, pos = tree_lines(tree, min[:], max_left, pos + 1, lines)
    
    min_right = min[:]
    min_right[fid] = thr
    lines, pos = tree_lines(tree, min_right, max[:], pos + 1, lines)
    
    return lines, pos
    
    
def dirty_set(S, p_outliers, outlier_weight):
    S = S.copy()
    n_outliers = int(S.shape[0] * p_outliers)
    indices = list(range(S.shape[0]))
    np.random.shuffle(indices)
    for i in indices[:n_outliers]:
        S[i] *= outlier_weight
    return S

def make_cov(n_features):
    colour = np.random.normal(size=(n_features, n_features))
    return np.dot(colour.T, colour)

def log_det_cov(S):
    return np.log(np.linalg.det(np.cov(S, rowvar=0, ddof=1)))

if __name__ == "__main__":
    main()
