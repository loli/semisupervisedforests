import numpy
from sklearn.covariance import MinCovDet, empirical_covariance, ledoit_wolf

mean0 = (0,0)
cov0 = numpy.asarray([[ 1. ,  0.5],
                      [ 0.5,  1. ]])
                      
print 'real cov:'
print cov0            

# NUMPY
# draw from 0-centered multi-variate normal distribution
print '0-centered multi-variate normal distribution:'
X = numpy.random.multivariate_normal(mean0, cov0, 1000)
print numpy.cov(X, rowvar=0, ddof=1) # should be ~cov0
print numpy.mean(X, axis=0) # should be ~mean0

mean1 = (3,6)
# draw from non-0-centered multi-variate normal distribution
print 'non-0-centered multi-variate normal distribution:'
X = numpy.random.multivariate_normal(mean1, cov0, 1000)
print numpy.cov(X, rowvar=0, ddof=1) # should be ~cov0
print numpy.mean(X, axis=0) # should be ~mean1

print '...its pre-centered covariance matric:'
print numpy.cov(X - numpy.mean(X, axis=0), rowvar=0, ddof=1) # same results as for as non-centered (allclose)

print '...and its correlation matrix:'
print numpy.corrcoef(X, rowvar=0, ddof=1)

# SKLEARN
print 'Empirical covariance:'
print empirical_covariance(X, assume_centered=False) # should be +/- same as numpy.cov

print 'Shrinkage covariance (Leodid-Wolf):'
print ledoit_wolf(X, assume_centered=False)    

print 'Minimum Covariance Determinant (MCD): robust estimator of covariance in non-multimodal case'
print 'From 10 samples:'
X = numpy.random.multivariate_normal(mean1, cov0, 10)
obj = MinCovDet()
obj.fit(X)
print obj.covariance_
print obj.error_norm(cov0)

print 'From 100 samples:'
X = numpy.random.multivariate_normal(mean1, cov0, 100)
obj = MinCovDet()
obj.fit(X)
print obj.covariance_
print obj.error_norm(cov0)

print 'From 1000 samples:'
X = numpy.random.multivariate_normal(mean1, cov0, 1000)
obj = MinCovDet()
obj.fit(X)
print obj.covariance_
print obj.error_norm(cov0)

print 'From 10000 samples:'
X = numpy.random.multivariate_normal(mean1, cov0, 10000)
obj = MinCovDet()
obj.fit(X)
print obj.covariance_
print obj.error_norm(cov0)

# DET(COV(X))

# LOG-likelihood of DET(COV(X))


