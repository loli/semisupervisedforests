Covariance matrix
=================
(also dispersion matrix or variance–covariance matrix)

Intuitively, the covariance matrix generalizes the notion of variance to multiple dimensions.
It is computed by computing the i,j covariances between all entries of X, AFTER the mean \mu(X) has been substracted from them.

Every covariance matrix is symmetric. In addition, every covariance matrix is positive semi-definite.

Empirical covariance matrix
---------------------------
(also statistical covariance matrix or maximum likelihood estimator)

Provided the number of observations is large enough compared to the number of features, this classical approach is an unbiased estimator of the corresponding population covariance matrix.

Sklearn says, that the resulting matrixes might be numerically unstable and should be "shrunken": http://scikit-learn.org/stable/modules/covariance.html#shrunk-covariance

Shrinkage of empirical covariance matrix
----------------------------------------
Sklearn propose (and implements) Ledoit-Wolf as well as Oracle Approximating shrinkage, which both return a biased, but numerically more stable version of the empirical covariance matrix.

[Maybe I should use them? But sensitive to outliers!]

Outlier treatment
-----------------
Sklearn claims that both, the direct empirical covariance matrix and the shrinkage approach are very sensitive to outliers. The Minimum Covariance Determinant (MCD) is proposed as more robust method (see below).

[I could employ this, rather than the empirical co-variance (as I surely face outliers). The question if, if this is not too robust against outliers i.e. does lead to undesired splits.]

numpy.cov
---------
This method internally centers the data matrices, independently of the bias/ddof supplied. [I tested this.]
For bias/ddof, the rule of thumb is:
    If the mean used for centering has been extracted from the data itself (i.e. estimated), use 1/(n-1) i.e. ddof=1. [my case]
    If the real mean of the underlying distribution has been used for centering (in my case unknown), use 1/n i.e. ddof=0.

Correlation matrix
==================
The correlation matrix can be seen as the covariance matrix of the standardized random variables.
Standardization is done by substracting the mean \mu(X) and dividing through the standard deviation \sigma(X).

The principal diagonal of the correlation matrix is always 1, each off-diagonal element is between 1 and -1 inclusive.

[Might this be a more stable version of the thing I am looking for? Or does the standardization remove required information?]

Determinant of covariance matrix
================================
Interpretation: Intuitively, gives a measure of how well the data points fit a multivariate normal distribution (i.e. how elliptically they are distributed).

Determinant
-----------
Interpretation (geometric): The absolute value of the determinant gives the scale factor by which area or volume (or a higher-dimensional analogue) is multiplied under the associated linear transformation, while its sign indicates whether the transformation preserves orientation. Thus a 2 × 2 matrix with determinant −2, when applied to a region of the plane with finite area, will transform that region into one with twice the area, while reversing its orientation.
Interpretation: If the determinant is nonzero, the matrix is nonsingular i.e. a unique solution to its linear equation exists.
Determinants are only defined for square matrices.

If det(A) = 0, A is singular.
If det(A) = +/-1, A is unimodular.
If in a matrix, any row or column is 0, then the determinant of that particular matrix is 0.
This n-linear function is an alternating form. This means that whenever two columns of a matrix are identical, or more generally some column can be expressed as a linear combination of the other columns (i.e. the columns of the matrix form a linearly dependent set), its determinant is 0.

Log Likelihood
==============
log(f(x)) is the log-likelihood of a function f(x) (log = natural logarithm i.e. base e)

log(x), x\in[0,1] \in (-\inf, 0]
log(x), x\in[1, +\inf) \in [0, +\inf)

derivative: 1/x

Log Likelihood of determinant of covariance matrix
==================================================

Computational hazards
---------------------
- If cov(X) is singular (i.e. det(cov(X)) == 0)), we can not compute log(0), since it's -\inf.

Range
-----

Multimodal data
===============
Observations drawn from a multimodal distribution i.e. a distribution with multiple peaks e.g. a mixed-gaussian.

Minimum Covariance Determinant (MCD)
====================================
(http://scikit-learn.org/stable/modules/covariance.html#robust-covariance-estimation)
Implemented in sklearn.
Could serve as an cov estimator for huge datasets.
Not robust against multimodal data.
Can cope with up to (n_observations - n_features - 1)/2 outliers.
Require ~1.000 to 10.000 samples for a stable estimate (my tests).
