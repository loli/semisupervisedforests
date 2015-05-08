# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False

# Authors: Gilles Louppe <g.louppe@gmail.com>
#          Peter Prettenhofer <peter.prettenhofer@gmail.com>
#          Brian Holt <bdholt1@gmail.com>
#          Noel Dawe <noel@dawe.me>
#          Satrajit Gosh <satrajit.ghosh@gmail.com>
#          Lars Buitinck <L.J.Buitinck@uva.nl>
#          Arnaud Joly <arnaud.v.joly@gmail.com>
#          Joel Nothman <joel.nothman@gmail.com>
#          Fares Hedayati <fares.hedayati@gmail.com>
#
# Licence: BSD 3 clause

from libc.stdlib cimport calloc, free, realloc, qsort

from libc.string cimport memcpy, memset
from libc.math cimport log as ln
from cpython cimport Py_INCREF, PyObject

import numpy as np
cimport numpy as np
np.import_array()

from scipy.sparse import issparse, csc_matrix, csr_matrix

from sklearn.tree._utils cimport Stack, StackRecord
from sklearn.tree._utils cimport PriorityHeap, PriorityHeapRecord



cdef extern from "numpy/arrayobject.h":
    object PyArray_NewFromDescr(object subtype, np.dtype descr,
                                int nd, np.npy_intp* dims,
                                np.npy_intp* strides,
                                void* data, int flags, object obj)

# =============================================================================
# Types and constants
# =============================================================================

from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE

cdef double INFINITY = np.inf
TREE_LEAF = -1
TREE_UNDEFINED = -2
cdef SIZE_t _TREE_LEAF = TREE_LEAF
cdef SIZE_t _TREE_UNDEFINED = TREE_UNDEFINED
cdef SIZE_t INITIAL_STACK_SIZE = 10

cdef DTYPE_t MIN_IMPURITY_SPLIT = 1e-7

# Mitigate precision differences between 32 bit and 64 bit
cdef DTYPE_t FEATURE_THRESHOLD = 1e-7

# Constant to switch between algorithm non zero value extract algorithm
# in SparseSplitter
cdef DTYPE_t EXTRACT_NNZ_SWITCH = 0.1

# Some handy constants (BestFirstTreeBuilder)
cdef int IS_FIRST = 1
cdef int IS_NOT_FIRST = 0
cdef int IS_LEFT = 1
cdef int IS_NOT_LEFT = 0

cdef enum:
    # Max value for our rand_r replacement (near the bottom).
    # We don't use RAND_MAX because it's different across platforms and
    # particularly tiny on Windows/MSVC.
    RAND_R_MAX = 0x7FFFFFFF

# Repeat struct definition for numpy
NODE_DTYPE = np.dtype({
    'names': ['left_child', 'right_child', 'feature', 'threshold', 'impurity',
              'n_node_samples', 'weighted_n_node_samples'],
    'formats': [np.intp, np.intp, np.intp, np.float64, np.float64, np.intp,
                np.float64],
    'offsets': [
        <Py_ssize_t> &(<Node*> NULL).left_child,
        <Py_ssize_t> &(<Node*> NULL).right_child,
        <Py_ssize_t> &(<Node*> NULL).feature,
        <Py_ssize_t> &(<Node*> NULL).threshold,
        <Py_ssize_t> &(<Node*> NULL).impurity,
        <Py_ssize_t> &(<Node*> NULL).n_node_samples,
        <Py_ssize_t> &(<Node*> NULL).weighted_n_node_samples
    ]
})


# =============================================================================
# Criterion
# =============================================================================

cdef class Criterion:
    """Interface for impurity criteria."""

    cdef void init(self, DOUBLE_t* y, SIZE_t y_stride, DOUBLE_t* sample_weight,
                   double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                   SIZE_t end) nogil:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end]."""
        pass

    cdef void reset(self) nogil:
        """Reset the criterion at pos=start."""
        pass

    cdef void update(self, SIZE_t new_pos) nogil:
        """Update the collected statistics by moving samples[pos:new_pos] from
           the right child to the left child."""
        pass

    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]."""
        pass

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of
           samples[start:pos] + the impurity of samples[pos:end]."""
        pass

    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] into dest."""
        pass

    cdef double impurity_improvement(self, double impurity) nogil:
        """Weighted impurity improvement, i.e.

           N_t / N * (impurity - N_t_L / N_t * left impurity
                               - N_t_L / N_t * right impurity),

           where N is the total number of samples, N_t is the number of samples
           in the current node, N_t_L is the number of samples in the left
           child and N_t_R is the number of samples in the right child."""
        cdef double impurity_left
        cdef double impurity_right

        self.children_impurity(&impurity_left, &impurity_right)

        #!TODO: self.weighted_n_samples contains also unlabeled samples in the unlabeled use-case! (set in Splitter)
        return ((self.weighted_n_node_samples / self.weighted_n_samples) *
                (impurity - self.weighted_n_right / self.weighted_n_node_samples * impurity_right
                          - self.weighted_n_left / self.weighted_n_node_samples * impurity_left))


cdef class ClassificationCriterion(Criterion):
    """Abstract criterion for classification."""
    cdef SIZE_t* n_classes
    cdef SIZE_t label_count_stride
    cdef double* label_count_left
    cdef double* label_count_right
    cdef double* label_count_total

    def __cinit__(self, SIZE_t n_outputs,
                  np.ndarray[SIZE_t, ndim=1] n_classes):
        # Default values
        self.y = NULL
        self.y_stride = 0
        self.sample_weight = NULL

        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        self.label_count_left = NULL
        self.label_count_right = NULL
        self.label_count_total = NULL

        # Count labels for each output
        self.n_classes = NULL
        safe_realloc(&self.n_classes, n_outputs)

        cdef SIZE_t k = 0
        cdef SIZE_t label_count_stride = 0

        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]

            if n_classes[k] > label_count_stride:
                label_count_stride = n_classes[k]

        self.label_count_stride = label_count_stride

        # Allocate counters
        cdef SIZE_t n_elements = n_outputs * label_count_stride
        self.label_count_left = <double*> calloc(n_elements, sizeof(double))
        self.label_count_right = <double*> calloc(n_elements, sizeof(double))
        self.label_count_total = <double*> calloc(n_elements, sizeof(double))

        # Check for allocation errors
        if (self.label_count_left == NULL or
                self.label_count_right == NULL or
                self.label_count_total == NULL):
            raise MemoryError()

    def __dealloc__(self):
        """Destructor."""
        free(self.n_classes)
        free(self.label_count_left)
        free(self.label_count_right)
        free(self.label_count_total)

    def __reduce__(self):
        return (ClassificationCriterion,
                (self.n_outputs,
                 sizet_ptr_to_ndarray(self.n_classes, self.n_outputs)),
                self.__getstate__())

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef void init(self, DOUBLE_t* y, SIZE_t y_stride,
                   DOUBLE_t* sample_weight, double weighted_n_samples,
                   SIZE_t* samples, SIZE_t start, SIZE_t end) nogil:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end]."""
        # Initialize fields
        self.y = y
        self.y_stride = y_stride
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        cdef double weighted_n_node_samples = 0.0

        # Initialize label_count_total and weighted_n_node_samples
        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef double* label_count_total = self.label_count_total

        cdef SIZE_t i = 0
        cdef SIZE_t p = 0
        cdef SIZE_t k = 0
        cdef SIZE_t c = 0
        cdef DOUBLE_t w = 1.0
        cdef SIZE_t offset = 0

        for k in range(n_outputs):
            memset(label_count_total + offset, 0,
                   n_classes[k] * sizeof(double))
            offset += label_count_stride

        for p in range(start, end):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(n_outputs):
                c = <SIZE_t> y[i * y_stride + k]
                label_count_total[k * label_count_stride + c] += w

            weighted_n_node_samples += w

        self.weighted_n_node_samples = weighted_n_node_samples

        # Reset to pos=start
        self.reset()

    cdef void reset(self) nogil:
        """Reset the criterion at pos=start."""
        self.pos = self.start

        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples

        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef double* label_count_total = self.label_count_total
        cdef double* label_count_left = self.label_count_left
        cdef double* label_count_right = self.label_count_right

        cdef SIZE_t k = 0

        for k in range(n_outputs):
            memset(label_count_left, 0, n_classes[k] * sizeof(double))
            memcpy(label_count_right, label_count_total,
                   n_classes[k] * sizeof(double))

            label_count_total += label_count_stride
            label_count_left += label_count_stride
            label_count_right += label_count_stride

    cdef void update(self, SIZE_t new_pos) nogil:
        """Update the collected statistics by moving samples[pos:new_pos] from
            the right child to the left child."""
        cdef DOUBLE_t* y = self.y
        cdef SIZE_t y_stride = self.y_stride
        cdef DOUBLE_t* sample_weight = self.sample_weight

        cdef SIZE_t* samples = self.samples
        cdef SIZE_t pos = self.pos

        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef double* label_count_total = self.label_count_total
        cdef double* label_count_left = self.label_count_left
        cdef double* label_count_right = self.label_count_right

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef SIZE_t label_index
        cdef DOUBLE_t w = 1.0
        cdef DOUBLE_t diff_w = 0.0

        # Note: We assume start <= pos < new_pos <= end

        for p in range(pos, new_pos):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(n_outputs):
                label_index = (k * label_count_stride +
                               <SIZE_t> y[i * y_stride + k])
                label_count_left[label_index] += w
                label_count_right[label_index] -= w

            diff_w += w

        self.weighted_n_left += diff_w
        self.weighted_n_right -= diff_w

        self.pos = new_pos

    cdef double node_impurity(self) nogil:
        pass

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        pass

    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] into dest."""
        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef double* label_count_total = self.label_count_total
        cdef SIZE_t k

        for k in range(n_outputs):
            memcpy(dest, label_count_total, n_classes[k] * sizeof(double))
            dest += label_count_stride
            label_count_total += label_count_stride


cdef class UnsupervisedClassificationCriterion(Criterion):
    """Criterion for un-supervised classification using differential entropy."""
    
    # note: sample weights can not be incorporated, assuming equal weight!
    # but: it could be introduced using weighted sample sums in the impurity_improvement method
    
    # create a copy X' of X
    # always work on memory of X' defined by start - end
    # upon init, place observatins/samples from X into X' according to sample-indices in `samples`, they will be ordered
    # for a split, simply define X_left as the X' from start to pos and X_right as the X' from pos to end
    # for the impurity calculation, compute log(det(\Sigma))...etc on X_left and X_right
    # advanced: try update log(det(\Sigma)) dynamically by moving samples from X_right to X_left

    def __cinit__(self, SIZE_t n_samples, SIZE_t n_features):
        # Default values
        self.X = NULL # all training samples
        self.X_stride = 0 # stride (i.e. feature length * feature_dtype_length)
        self.S = NULL # current node's set of samples, ordered to contain S_left and S_right between start, pos and end
        self.sample_weight = NULL # sample weight in 1-D double array

        self.samples = NULL # sample ids, ordered by splitter between start and end
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_samples = n_samples # X.shape[0]
        self.n_features = n_features # X.shape[1]
        self.n_node_samples = 0 # size of S
        self.weighted_n_node_samples = 0.0 # total weight of all node samples (all in S)
        self.weighted_n_left = 0.0 # weight of all samples in S_left
        self.weighted_n_right = 0.0 # weight of all samples in S_right       
        
        # Allocate memory
        self.S = <DTYPE_t*> calloc(n_samples * n_features, sizeof(DTYPE_t)) # same size as X

        # Check for allocation errors
        if self.S == NULL:
            raise MemoryError()

    def __dealloc__(self):
        """Destructor."""
        free(self.S)
        
    def __reduce__(self): # arguemnts to __cinit__
        return (UnsupervisedClassificationCriterion,
                (self.n_samples,
                 self.n_features),
                 self.__getstate__())

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef void sortS(self) nogil:
        """Re-sort the sample set S according to sample-id set samples."""
        cdef DTYPE_t* X = self.X
        cdef DTYPE_t* S = self.S
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t X_stride = self.X_stride
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end
        cdef SIZE_t n_node_samples = self.n_node_samples
        
        cdef SIZE_t i = 0
        
        #!TODO: make this a sort, not a re-copy
        #!TODO: Possible bug: I think, that samples[i] should actually be samples[start + i], as that part of samples is currently in use
        #       To check this, I must output the results S_left and S_right sets at deeper levels (e.g 2)
        for i in range(n_node_samples):
            memcpy(S + (start + i) * X_stride,
                   X + samples[i + start] * X_stride,
                   X_stride * sizeof(DTYPE_t))
            
        #with gil:
        #    print '### SORTS ###'
        

    cdef void init2(self, DTYPE_t* X, SIZE_t X_stride,
                    DOUBLE_t* sample_weight, double weighted_n_samples,
                    SIZE_t* samples, SIZE_t start, SIZE_t end) nogil:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end].
           Also sorts S accordingly.
           Note that the signature of this init2() does not
           correspond to the parent classes (Criterion) init()
           signature and therefore can only be used by a dedicated
           splitter class."""
        # Initialize fields
        self.X = X
        self.X_stride = X_stride
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        
        # initialize weighted_n_node_samples
        cdef SIZE_t n_node_samples = self.n_node_samples
        cdef double weighted_n_node_samples = 0.0
        cdef SIZE_t i = 0
        cdef SIZE_t p = 0
        
        #!TODO: What is with zero-weight samples? Are they already removed from "samples"
        #       or would I have to take care of this here? After all, they should not figure
        #       in the entropy computation.
        #       This is anyway a problem, as the weight figure in the information gain, but not
        #       directly in the entropy computation -> could this be a problem?
        
        if sample_weight == NULL:
            weighted_n_node_samples = 1.0 * n_node_samples
        else:
            for p in range(start, end):
                i = samples[p]
                weighted_n_node_samples += sample_weight[i]

        self.weighted_n_node_samples = weighted_n_node_samples

        # sort samples set S
        self.sortS()

        # Reset to pos=start
        self.reset()

    cdef void reset(self) nogil:
        """Reset the criterion at pos=start."""
        self.pos = self.start
        
        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples

    cdef void update(self, SIZE_t new_pos) nogil:
        """Update the collected statistics by moving samples[pos:new_pos] from
            the right child to the left child.""" 
        cdef DOUBLE_t* sample_weight = self.sample_weight
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t pos = self.pos
        
        cdef SIZE_t i
        cdef SIZE_t p
        cdef DOUBLE_t diff_w = 0.0        
        
        if sample_weight == NULL:
            diff_w = 1.0 * (new_pos - pos)
        else:
            for p in range(pos, new_pos):
                i = samples[p]
                diff_w += sample_weight[i]
            
        self.weighted_n_left += diff_w
        self.weighted_n_right -= diff_w
        
        self.pos = new_pos

    cdef double node_impurity(self) nogil:
        """Compute the impurity of the current node."""
        cdef SIZE_t start = self.start
        cdef SIZE_t X_stride = self.X_stride
        cdef SIZE_t n_node_samples = self.n_node_samples
        cdef double entropy = 0.0
        cdef DTYPE_t* S = self.S
        
        #!TODO: Remove the "with gil:" here, when differential_entropy finally nogil function
        with gil:
            entropy = self.differential_entropy(S + start * X_stride, n_node_samples) # src, n_samples
        
        return entropy

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        cdef SIZE_t start = self.start
        cdef SIZE_t pos = self.pos
        cdef SIZE_t end = self.end
        cdef SIZE_t X_stride = self.X_stride
        cdef DTYPE_t* S = self.S
        
        #!TODO: Remove the "with gil:" here, when differential_entropy finally nogil function
        with gil:
            impurity_left[0] = self.differential_entropy(S + start * X_stride, pos - start)
            impurity_right[0] = self.differential_entropy(S + pos * X_stride, end - pos)

    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] into dest."""
        #!TODO: Here the variables of the fitted multivariate Gaussian distribution
        #       represented by this leaf node should be written into the given memory.
        # I must figure out    1. how many variables this will be and
        #                      2. how to ensure that the tree provides the required size (n_classes/output * n_outputs * double)
        
        # memory space provide:
        #     [n_outputs, max_n_classes] of double
        # original purpose:
        #     Contains the constant prediction value of each node.
        # I require:
        #     n_features + n_features^2 + 1 of double
        #     mu           cov            frac/Z
        
        # Computation order:
        #    Here I can calculate the cov, mu and frac
        #    Afterwards, I'll have to come back to the tree and calculate
        #    partition funtion Z and replace frac by  frac/Z for each node
        
        # !TODO: replace with nogil version
        # !TODO: Very crude version!
        # convert memory block to numpy array
        cdef SIZE_t X_stride = self.X_stride
        cdef SIZE_t start = self.start
        cdef DTYPE_t* S = self.S
        cdef SIZE_t n_node_samples = self.n_node_samples
        cdef DTYPE_t [:,::1] arr_view
        
        cdef double [:, ::1] mcov
        cdef double [::1] mmu
        
        cdef SIZE_t i = 0
        cdef SIZE_t p = self.n_samples
        
        #with gil:
        #    print '### NODE_VALUE ###'
        #    print 'start/pos/end', self.start, self.pos, self.end
        #    print 'n_node_samples/end-start', self.n_node_samples, self.end - self.start
        #    print 'n_samples', self.n_samples
        #    print 'S:'
        #    for i in range(p):
        #        print i, self.S[i*2], self.S[i*2+1]
              
        with gil:
            arr_view = <DTYPE_t[:n_node_samples,:X_stride]> &S[start * X_stride]
            arr = np.asarray(arr_view).copy().astype(np.float64)
            #print '-----------------'
            #print 'S[0]', S[0]
            #print 'self.S[0]', self.S[0]
            #print 'S[1]', S[1]
            #print 'self.S[1]', self.S[1]
            #print 'start * X_stride', start * X_stride
            #print 'self.S[start * X_stride]', self.S[start * X_stride]
            #print '-----------------'
            #print 'arr.shape', arr.shape
            #print 'arr:'
            #print arr

            cov = np.cov(arr, rowvar=0, ddof=1)
            mu = np.mean(arr, axis=0)
            frac = self.weighted_n_node_samples / self.weighted_n_samples
            
            mcov = cov
            mmu = mu
            
            #print '-----------------'
            #print 'frac:', frac
            #print 'cov:'
            #print cov
            #print 'mu:'
            #print mu
        
        #!TODO: Does not work, see [...].tree_.value
        memcpy(dest, &frac, sizeof(double))
        dest += 1
        memcpy(dest, &mcov[0,0], self.n_features * self.n_features * sizeof(double))
        dest += self.n_features * self.n_features
        memcpy(dest, &mmu[0], self.n_features * sizeof(double))

    cdef double differential_entropy(self, DTYPE_t* src, SIZE_t size):
        """Compute the differential entropy (also called continuous- or unsupervised-),
        which is defined as log(det(cov(S))) of a set S of observations identified
        through src and size."""
        cdef SIZE_t X_stride = self.X_stride
        
        #!TODO: employing np.log(MIN_IMPURITY_SPLIT) as a minimal value
        # is insecure, if <10 samples are in the parent node. Min_samples_leaf
        # should be used or the depth restricted or such.
        # The same solution is used below when det < MIN_IMPURITY_SPLIT,
        # which is equally questionable
        
        # skip if set size <= 1
        if size <= 1: return np.log(MIN_IMPURITY_SPLIT)

        # convert memory block to numpy array
        cdef DTYPE_t [:,::1] arr_view = <DTYPE_t[:size,:X_stride]> src
        cdef np.ndarray arr = np.asarray(arr_view).copy()
        
        #!TODO: Place here my dynstatcov and make function nogil
        # compute the differential entropy using numpy
        cov = np.cov(arr, rowvar=0, ddof=1)
        det = np.linalg.det(cov)
        if det < 0: det *= -1
        if det < MIN_IMPURITY_SPLIT: det = MIN_IMPURITY_SPLIT
        return np.log(det)


cdef class Entropy(ClassificationCriterion):
    """Cross Entropy impurity criteria.

    Let the target be a classification outcome taking values in 0, 1, ..., K-1.
    If node m represents a region Rm with Nm observations, then let

        pmk = 1/ Nm \sum_{x_i in Rm} I(yi = k)

    be the proportion of class k observations in node m.

    The cross-entropy is then defined as

        cross-entropy = - \sum_{k=0}^{K-1} pmk log(pmk)
    """
    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]."""
        cdef double weighted_n_node_samples = self.weighted_n_node_samples

        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef double* label_count_total = self.label_count_total

        cdef double entropy = 0.0
        cdef double total = 0.0
        cdef double tmp
        cdef SIZE_t k
        cdef SIZE_t c

        for k in range(n_outputs):
            entropy = 0.0

            for c in range(n_classes[k]):
                tmp = label_count_total[c]
                if tmp > 0.0:
                    tmp /= weighted_n_node_samples
                    entropy -= tmp * log(tmp)

            total += entropy
            label_count_total += label_count_stride

        return total / n_outputs

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (samples[start:pos]) and the impurity the right child
           (samples[pos:end])."""
        cdef double weighted_n_node_samples = self.weighted_n_node_samples
        cdef double weighted_n_left = self.weighted_n_left
        cdef double weighted_n_right = self.weighted_n_right

        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef double* label_count_left = self.label_count_left
        cdef double* label_count_right = self.label_count_right

        cdef double entropy_left = 0.0
        cdef double entropy_right = 0.0
        cdef double total_left = 0.0
        cdef double total_right = 0.0
        cdef double tmp
        cdef SIZE_t k
        cdef SIZE_t c

        for k in range(n_outputs):
            entropy_left = 0.0
            entropy_right = 0.0

            for c in range(n_classes[k]):
                tmp = label_count_left[c]
                if tmp > 0.0:
                    tmp /= weighted_n_left
                    entropy_left -= tmp * log(tmp)

                tmp = label_count_right[c]
                if tmp > 0.0:
                    tmp /= weighted_n_right
                    entropy_right -= tmp * log(tmp)

            total_left += entropy_left
            total_right += entropy_right
            label_count_left += label_count_stride
            label_count_right += label_count_stride

        impurity_left[0] = total_left / n_outputs
        impurity_right[0] = total_right / n_outputs

cdef class LabeledOnlyEntropy(Entropy):
    """Cross Entropy impurity criteria applied to labeled samples only."""

    cdef void init(self, DOUBLE_t* y, SIZE_t y_stride,
                   DOUBLE_t* sample_weight, double weighted_n_samples,
                   SIZE_t* samples, SIZE_t start, SIZE_t end) nogil:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end]."""
        # Initialize fields
        self.y = y
        self.y_stride = y_stride
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        cdef double weighted_n_node_samples = 0.0

        # Initialize label_count_total and weighted_n_node_samples
        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef double* label_count_total = self.label_count_total

        cdef SIZE_t i = 0
        cdef SIZE_t p = 0
        cdef SIZE_t k = 0
        cdef SIZE_t c = 0
        cdef DOUBLE_t w = 1.0
        cdef SIZE_t offset = 0

        for k in range(n_outputs):
            memset(label_count_total + offset, 0,
                   n_classes[k] * sizeof(double))
            offset += label_count_stride

        for p in range(start, end):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            c = <SIZE_t> y[i * y_stride]
            if c > 0: # don't consider unlabeled samples; assuming that sample is either for all outputs unlabeled or for all labeled
                label_count_total[k * label_count_stride + c - 1] += w
                
                for k in range(1, n_outputs):
                    c = <SIZE_t> y[i * y_stride + k]
                    label_count_total[k * label_count_stride + c - 1] += w
                
                weighted_n_node_samples += w
        self.weighted_n_node_samples = weighted_n_node_samples

        # Reset to pos=start
        self.reset()

    cdef void update(self, SIZE_t new_pos) nogil:
        """Update the collected statistics by moving samples[pos:new_pos] from
            the right child to the left child."""
        cdef DOUBLE_t* y = self.y
        cdef SIZE_t y_stride = self.y_stride
        cdef DOUBLE_t* sample_weight = self.sample_weight

        cdef SIZE_t* samples = self.samples
        cdef SIZE_t pos = self.pos

        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef double* label_count_total = self.label_count_total
        cdef double* label_count_left = self.label_count_left
        cdef double* label_count_right = self.label_count_right

        cdef SIZE_t i = 0
        cdef SIZE_t p = 0
        cdef SIZE_t c = 0
        cdef SIZE_t k = 0
        cdef SIZE_t label_index
        cdef DOUBLE_t w = 1.0
        cdef DOUBLE_t diff_w = 0.0

        # Note: We assume start <= pos < new_pos <= end

        for p in range(pos, new_pos):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            c = <SIZE_t> y[i * y_stride]
            if c > 0: # don't consider unlabeled samples; assuming that sample is either for all outputs unlabeled or for all labeled
                label_index = k * label_count_stride + c
                label_count_left[label_index] += w
                label_count_right[label_index] -= w
                
                for k in range(1, n_outputs):
                    label_index = (k * label_count_stride +
                                   <SIZE_t> y[i * y_stride + k])
                    label_count_left[label_index] += w
                    label_count_right[label_index] -= w
    
                diff_w += w

        self.weighted_n_left += diff_w
        self.weighted_n_right -= diff_w

        self.pos = new_pos   

cdef class Gini(ClassificationCriterion):
    """Gini Index impurity criteria.

    Let the target be a classification outcome taking values in 0, 1, ..., K-1.
    If node m represents a region Rm with Nm observations, then let

        pmk = 1/ Nm \sum_{x_i in Rm} I(yi = k)

    be the proportion of class k observations in node m.

    The Gini Index is then defined as:

        index = \sum_{k=0}^{K-1} pmk (1 - pmk)
              = 1 - \sum_{k=0}^{K-1} pmk ** 2
    """
    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]."""
        cdef double weighted_n_node_samples = self.weighted_n_node_samples

        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef double* label_count_total = self.label_count_total

        cdef double gini = 0.0
        cdef double total = 0.0
        cdef double tmp
        cdef SIZE_t k
        cdef SIZE_t c

        for k in range(n_outputs):
            gini = 0.0

            for c in range(n_classes[k]):
                tmp = label_count_total[c]
                gini += tmp * tmp

            gini = 1.0 - gini / (weighted_n_node_samples *
                                 weighted_n_node_samples)

            total += gini
            label_count_total += label_count_stride

        return total / n_outputs

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (samples[start:pos]) and the impurity the right child
           (samples[pos:end])."""
        cdef double weighted_n_node_samples = self.weighted_n_node_samples
        cdef double weighted_n_left = self.weighted_n_left
        cdef double weighted_n_right = self.weighted_n_right

        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t* n_classes = self.n_classes
        cdef SIZE_t label_count_stride = self.label_count_stride
        cdef double* label_count_left = self.label_count_left
        cdef double* label_count_right = self.label_count_right

        cdef double gini_left = 0.0
        cdef double gini_right = 0.0
        cdef double total = 0.0
        cdef double total_left = 0.0
        cdef double total_right = 0.0
        cdef double tmp
        cdef SIZE_t k
        cdef SIZE_t c

        for k in range(n_outputs):
            gini_left = 0.0
            gini_right = 0.0

            for c in range(n_classes[k]):
                tmp = label_count_left[c]
                gini_left += tmp * tmp
                tmp = label_count_right[c]
                gini_right += tmp * tmp

            gini_left = 1.0 - gini_left / (weighted_n_left *
                                           weighted_n_left)
            gini_right = 1.0 - gini_right / (weighted_n_right *
                                             weighted_n_right)

            total_left += gini_left
            total_right += gini_right
            label_count_left += label_count_stride
            label_count_right += label_count_stride

        impurity_left[0] = total_left / n_outputs
        impurity_right[0] = total_right / n_outputs


cdef class RegressionCriterion(Criterion):
    """Abstract criterion for regression.

    Computes variance of the target values left and right of the split point.
    Computation is linear in `n_samples` by using ::

        var = \sum_i^n (y_i - y_bar) ** 2
            = (\sum_i^n y_i ** 2) - n_samples y_bar ** 2
    """
    cdef double* mean_left
    cdef double* mean_right
    cdef double* mean_total
    cdef double* sq_sum_left
    cdef double* sq_sum_right
    cdef double* sq_sum_total
    cdef double* var_left
    cdef double* var_right
    cdef double* sum_left
    cdef double* sum_right
    cdef double* sum_total

    def __cinit__(self, SIZE_t n_outputs):
        # Default values
        self.y = NULL
        self.y_stride = 0
        self.sample_weight = NULL

        self.samples = NULL
        self.start = 0
        self.pos = 0
        self.end = 0

        self.n_outputs = n_outputs
        self.n_node_samples = 0
        self.weighted_n_node_samples = 0.0
        self.weighted_n_left = 0.0
        self.weighted_n_right = 0.0

        # Allocate accumulators. Make sure they are NULL, not uninitialized,
        # before an exception can be raised (which triggers __dealloc__).
        self.mean_left = NULL
        self.mean_right = NULL
        self.mean_total = NULL
        self.sq_sum_left = NULL
        self.sq_sum_right = NULL
        self.sq_sum_total = NULL
        self.var_left = NULL
        self.var_right = NULL
        self.sum_left = NULL
        self.sum_right = NULL
        self.sum_total = NULL

        self.mean_left = <double*> calloc(n_outputs, sizeof(double))
        self.mean_right = <double*> calloc(n_outputs, sizeof(double))
        self.mean_total = <double*> calloc(n_outputs, sizeof(double))
        self.sq_sum_left = <double*> calloc(n_outputs, sizeof(double))
        self.sq_sum_right = <double*> calloc(n_outputs, sizeof(double))
        self.sq_sum_total = <double*> calloc(n_outputs, sizeof(double))
        self.var_left = <double*> calloc(n_outputs, sizeof(double))
        self.var_right = <double*> calloc(n_outputs, sizeof(double))
        self.sum_left = <double*> calloc(n_outputs, sizeof(double))
        self.sum_right = <double*> calloc(n_outputs, sizeof(double))
        self.sum_total = <double*> calloc(n_outputs, sizeof(double))

        if (self.mean_left == NULL or
                self.mean_right == NULL or
                self.mean_total == NULL or
                self.sq_sum_left == NULL or
                self.sq_sum_right == NULL or
                self.sq_sum_total == NULL or
                self.var_left == NULL or
                self.var_right == NULL or
                self.sum_left == NULL or
                self.sum_right == NULL or
                self.sum_total == NULL):
            raise MemoryError()

    def __dealloc__(self):
        """Destructor."""
        free(self.mean_left)
        free(self.mean_right)
        free(self.mean_total)
        free(self.sq_sum_left)
        free(self.sq_sum_right)
        free(self.sq_sum_total)
        free(self.var_left)
        free(self.var_right)
        free(self.sum_left)
        free(self.sum_right)
        free(self.sum_total)

    def __reduce__(self):
        return (RegressionCriterion, (self.n_outputs,), self.__getstate__())

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef void init(self, DOUBLE_t* y, SIZE_t y_stride, DOUBLE_t* sample_weight,
                   double weighted_n_samples, SIZE_t* samples, SIZE_t start,
                   SIZE_t end) nogil:
        """Initialize the criterion at node samples[start:end] and
           children samples[start:start] and samples[start:end]."""
        # Initialize fields
        self.y = y
        self.y_stride = y_stride
        self.sample_weight = sample_weight
        self.samples = samples
        self.start = start
        self.end = end
        self.n_node_samples = end - start
        self.weighted_n_samples = weighted_n_samples
        cdef double weighted_n_node_samples = 0.

        # Initialize accumulators
        cdef SIZE_t n_outputs = self.n_outputs
        cdef double* mean_left = self.mean_left
        cdef double* mean_right = self.mean_right
        cdef double* mean_total = self.mean_total
        cdef double* sq_sum_left = self.sq_sum_left
        cdef double* sq_sum_right = self.sq_sum_right
        cdef double* sq_sum_total = self.sq_sum_total
        cdef double* var_left = self.var_left
        cdef double* var_right = self.var_right
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double* sum_total = self.sum_total

        cdef SIZE_t i = 0
        cdef SIZE_t p = 0
        cdef SIZE_t k = 0
        cdef DOUBLE_t y_ik = 0.0
        cdef DOUBLE_t w_y_ik = 0.0
        cdef DOUBLE_t w = 1.0

        cdef SIZE_t n_bytes = n_outputs * sizeof(double)
        memset(mean_left, 0, n_bytes)
        memset(mean_right, 0, n_bytes)
        memset(mean_total, 0, n_bytes)
        memset(sq_sum_left, 0, n_bytes)
        memset(sq_sum_right, 0, n_bytes)
        memset(sq_sum_total, 0, n_bytes)
        memset(var_left, 0, n_bytes)
        memset(var_right, 0, n_bytes)
        memset(sum_left, 0, n_bytes)
        memset(sum_right, 0, n_bytes)
        memset(sum_total, 0, n_bytes)

        for p in range(start, end):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(n_outputs):
                y_ik = y[i * y_stride + k]
                w_y_ik = w * y_ik
                sum_total[k] += w_y_ik
                sq_sum_total[k] += w_y_ik * y_ik

            weighted_n_node_samples += w

        self.weighted_n_node_samples = weighted_n_node_samples

        for k in range(n_outputs):
            mean_total[k] = sum_total[k] / weighted_n_node_samples

        # Reset to pos=start
        self.reset()

    cdef void reset(self) nogil:
        """Reset the criterion at pos=start."""
        self.pos = self.start

        self.weighted_n_left = 0.0
        self.weighted_n_right = self.weighted_n_node_samples

        cdef SIZE_t n_outputs = self.n_outputs
        cdef double* mean_left = self.mean_left
        cdef double* mean_right = self.mean_right
        cdef double* mean_total = self.mean_total
        cdef double* sq_sum_left = self.sq_sum_left
        cdef double* sq_sum_right = self.sq_sum_right
        cdef double* sq_sum_total = self.sq_sum_total
        cdef double* var_left = self.var_left
        cdef double* var_right = self.var_right
        cdef double weighted_n_node_samples = self.weighted_n_node_samples
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double* sum_total = self.sum_total

        cdef SIZE_t k = 0

        for k in range(n_outputs):
            mean_right[k] = mean_total[k]
            mean_left[k] = 0.0
            sq_sum_right[k] = sq_sum_total[k]
            sq_sum_left[k] = 0.0
            var_right[k] = (sq_sum_right[k] / weighted_n_node_samples -
                            mean_right[k] * mean_right[k])
            var_left[k] = 0.0
            sum_right[k] = sum_total[k]
            sum_left[k] = 0.0

    cdef void update(self, SIZE_t new_pos) nogil:
        """Update the collected statistics by moving samples[pos:new_pos] from
           the right child to the left child."""
        cdef DOUBLE_t* y = self.y
        cdef SIZE_t y_stride = self.y_stride
        cdef DOUBLE_t* sample_weight = self.sample_weight

        cdef SIZE_t* samples = self.samples
        cdef SIZE_t pos = self.pos

        cdef SIZE_t n_outputs = self.n_outputs
        cdef double* mean_left = self.mean_left
        cdef double* mean_right = self.mean_right
        cdef double* sq_sum_left = self.sq_sum_left
        cdef double* sq_sum_right = self.sq_sum_right
        cdef double* var_left = self.var_left
        cdef double* var_right = self.var_right
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right

        cdef double weighted_n_left = self.weighted_n_left
        cdef double weighted_n_right = self.weighted_n_right

        cdef SIZE_t i
        cdef SIZE_t p
        cdef SIZE_t k
        cdef DOUBLE_t w = 1.0
        cdef DOUBLE_t diff_w = 0.0
        cdef DOUBLE_t y_ik, w_y_ik

        # Note: We assume start <= pos < new_pos <= end
        for p in range(pos, new_pos):
            i = samples[p]

            if sample_weight != NULL:
                w = sample_weight[i]

            for k in range(n_outputs):
                y_ik = y[i * y_stride + k]
                w_y_ik = w * y_ik

                sum_left[k] += w_y_ik
                sum_right[k] -= w_y_ik

                sq_sum_left[k] += w_y_ik * y_ik
                sq_sum_right[k] -= w_y_ik * y_ik

            diff_w += w

        weighted_n_left += diff_w
        weighted_n_right -= diff_w

        for k in range(n_outputs):
            mean_left[k] = sum_left[k] / weighted_n_left
            mean_right[k] = sum_right[k] / weighted_n_right
            var_left[k] = (sq_sum_left[k] / weighted_n_left -
                           mean_left[k] * mean_left[k])
            var_right[k] = (sq_sum_right[k] / weighted_n_right -
                            mean_right[k] * mean_right[k])

        self.weighted_n_left = weighted_n_left
        self.weighted_n_right = weighted_n_right

        self.pos = new_pos

    cdef double node_impurity(self) nogil:
        pass

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        pass

    cdef void node_value(self, double* dest) nogil:
        """Compute the node value of samples[start:end] into dest."""
        memcpy(dest, self.mean_total, self.n_outputs * sizeof(double))


cdef class MSE(RegressionCriterion):
    """Mean squared error impurity criterion.

        MSE = var_left + var_right
    """
    cdef double node_impurity(self) nogil:
        """Evaluate the impurity of the current node, i.e. the impurity of
           samples[start:end]."""
        cdef SIZE_t n_outputs = self.n_outputs
        cdef double* sq_sum_total = self.sq_sum_total
        cdef double* mean_total = self.mean_total
        cdef double weighted_n_node_samples = self.weighted_n_node_samples
        cdef double total = 0.0
        cdef SIZE_t k

        for k in range(n_outputs):
            total += (sq_sum_total[k] / weighted_n_node_samples -
                      mean_total[k] * mean_total[k])

        return total / n_outputs

    cdef void children_impurity(self, double* impurity_left,
                                double* impurity_right) nogil:
        """Evaluate the impurity in children nodes, i.e. the impurity of the
           left child (samples[start:pos]) and the impurity the right child
           (samples[pos:end])."""
        cdef SIZE_t n_outputs = self.n_outputs
        cdef double* var_left = self.var_left
        cdef double* var_right = self.var_right
        cdef double total_left = 0.0
        cdef double total_right = 0.0
        cdef SIZE_t k

        for k in range(n_outputs):
            total_left += var_left[k]
            total_right += var_right[k]

        impurity_left[0] = total_left / n_outputs
        impurity_right[0] = total_right / n_outputs


cdef class FriedmanMSE(MSE):
    """Mean squared error impurity criterion with improvement score by Friedman

    Uses the formula (35) in Friedmans original Gradient Boosting paper:

        diff = mean_left - mean_right
        improvement = n_left * n_right * diff^2 / (n_left + n_right)
    """

    cdef double impurity_improvement(self, double impurity) nogil:
        cdef SIZE_t n_outputs = self.n_outputs
        cdef SIZE_t k
        cdef double* sum_left = self.sum_left
        cdef double* sum_right = self.sum_right
        cdef double total_sum_left = 0.0
        cdef double total_sum_right = 0.0
        cdef double weighted_n_left = self.weighted_n_left
        cdef double weighted_n_right = self.weighted_n_right
        cdef double diff = 0.0

        for k from 0 <= k < n_outputs:
            total_sum_left += sum_left[k]
            total_sum_right += sum_right[k]

        total_sum_left = total_sum_left / n_outputs
        total_sum_right = total_sum_right / n_outputs
        diff = ((total_sum_left / weighted_n_left) -
                (total_sum_right / weighted_n_right))

        return (weighted_n_left * weighted_n_right * diff * diff /
                (weighted_n_left + weighted_n_right))

# =============================================================================
# Splitter
# =============================================================================

cdef inline void _init_split(SplitRecord* self, SIZE_t start_pos) nogil:
    self.impurity_left = INFINITY
    self.impurity_right = INFINITY
    self.pos = start_pos
    self.feature = 0
    self.threshold = 0.
    self.improvement = -INFINITY


cdef class Splitter:
    def __cinit__(self, Criterion criterion, SIZE_t max_features,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  object random_state):
        self.criterion = criterion

        self.samples = NULL
        self.n_samples = 0
        self.features = NULL
        self.n_features = 0
        self.feature_values = NULL

        self.y = NULL
        self.y_stride = 0
        self.sample_weight = NULL

        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.random_state = random_state

    def __dealloc__(self):
        """Destructor."""
        free(self.samples)
        free(self.features)
        free(self.constant_features)
        free(self.feature_values)

    def __getstate__(self):
        return {}

    def __setstate__(self, d):
        pass

    cdef void init(self,
                   object X,
                   np.ndarray[DOUBLE_t, ndim=2, mode="c"] y,
                   DOUBLE_t* sample_weight) except *:
        """Initialize the splitter."""
        # Reset random state
        self.rand_r_state = self.random_state.randint(0, RAND_R_MAX)
        # Initialize samples and features structures
        cdef SIZE_t n_samples = X.shape[0]
        cdef SIZE_t* samples = safe_realloc(&self.samples, n_samples)

        cdef SIZE_t i, j
        cdef double weighted_n_samples = 0.0
        j = 0

        for i in range(n_samples):
            # Only work with positively weighted samples
            if sample_weight == NULL or sample_weight[i] != 0.0:
                samples[j] = i
                j += 1

            if sample_weight != NULL:
                weighted_n_samples += sample_weight[i]
                #print sample_weight[i]
            else:
                weighted_n_samples += 1.0

        self.n_samples = j
        #print 'init:N_SAMPLES-posweight:', self.n_samples
        self.weighted_n_samples = weighted_n_samples
        #print 'init:weighted_n_samples:', self.weighted_n_samples

        cdef SIZE_t n_features = X.shape[1]
        cdef SIZE_t* features = safe_realloc(&self.features, n_features)

        for i in range(n_features):
            features[i] = i

        self.n_features = n_features

        safe_realloc(&self.feature_values, n_samples)
        safe_realloc(&self.constant_features, n_features)

        # Initialize y, sample_weight
        self.y = <DOUBLE_t*> y.data
        self.y_stride = <SIZE_t> y.strides[0] / <SIZE_t> y.itemsize
        self.sample_weight = sample_weight
        #print 'init:N_SAMPLES-methodend:', self.n_samples

    cdef void node_reset(self, SIZE_t start, SIZE_t end,
                         double* weighted_n_node_samples) nogil:
        """Reset splitter on node samples[start:end]."""
        self.start = start
        self.end = end

        self.criterion.init(self.y,
                            self.y_stride,
                            self.sample_weight,
                            self.weighted_n_samples,
                            self.samples,
                            start,
                            end)

        weighted_n_node_samples[0] = self.criterion.weighted_n_node_samples

    cdef void node_split(self, double impurity, SplitRecord* split,
                         SIZE_t* n_constant_features) nogil:
        """Find a split on node samples[start:end]."""
        pass

    cdef void node_value(self, double* dest) nogil:
        """Copy the value of node samples[start:end] into dest."""
        self.criterion.node_value(dest)

    cdef double node_impurity(self) nogil:
        """Copy the impurity of node samples[start:end."""
        return self.criterion.node_impurity()


cdef class BaseDenseSplitter(Splitter):
    cdef DTYPE_t* X
    cdef SIZE_t X_sample_stride
    cdef SIZE_t X_fx_stride

    def __cinit__(self, Criterion criterion, SIZE_t max_features,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  object random_state):
        # Parent __cinit__ is automatically called

        self.X = NULL
        self.X_sample_stride = 0
        self.X_fx_stride = 0

    cdef void init(self,
                   object X,
                   np.ndarray[DOUBLE_t, ndim=2, mode="c"] y,
                   DOUBLE_t* sample_weight) except *:
        """Initialize the splitter."""

        # Call parent init
        #print 'bds-init:pre-base-init:', self.n_samples
        Splitter.init(self, X, y, sample_weight)
        #print 'bds-init:post-base-init:', self.n_samples

        # Initialize X
        cdef np.ndarray X_ndarray = X

        self.X = <DTYPE_t*> X_ndarray.data
        self.X_sample_stride = <SIZE_t> X.strides[0] / <SIZE_t> X.itemsize
        self.X_fx_stride = <SIZE_t> X.strides[1] / <SIZE_t> X.itemsize
        #print 'bds-init:method-end:', self.n_samples


cdef class BestSplitter(BaseDenseSplitter):
    """Splitter for finding the best split."""
    def __reduce__(self):
        return (BestSplitter, (self.criterion,
                               self.max_features,
                               self.min_samples_leaf,
                               self.min_weight_leaf,
                               self.random_state), self.__getstate__())

    cdef void node_split(self, double impurity, SplitRecord* split,
                         SIZE_t* n_constant_features) nogil:
        """Find the best split on node samples[start:end]."""
        # Find the best split
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        cdef SIZE_t* features = self.features
        cdef SIZE_t* constant_features = self.constant_features
        cdef SIZE_t n_features = self.n_features

        cdef DTYPE_t* X = self.X
        cdef DTYPE_t* Xf = self.feature_values
        cdef SIZE_t X_sample_stride = self.X_sample_stride
        cdef SIZE_t X_fx_stride = self.X_fx_stride
        cdef SIZE_t max_features = self.max_features
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef SplitRecord best, current

        cdef SIZE_t f_i = n_features
        cdef SIZE_t f_j, p, tmp
        cdef SIZE_t n_visited_features = 0
        # Number of features discovered to be constant during the split search
        cdef SIZE_t n_found_constants = 0
        # Number of features known to be constant and drawn without replacement
        cdef SIZE_t n_drawn_constants = 0
        cdef SIZE_t n_known_constants = n_constant_features[0]
        # n_total_constants = n_known_constants + n_found_constants
        cdef SIZE_t n_total_constants = n_known_constants
        cdef DTYPE_t current_feature_value
        cdef SIZE_t partition_end

        _init_split(&best, end)

        # Sample up to max_features without replacement using a
        # Fisher-Yates-based algorithm (using the local variables `f_i` and
        # `f_j` to compute a permutation of the `features` array).
        #
        # Skip the CPU intensive evaluation of the impurity criterion for
        # features that were already detected as constant (hence not suitable
        # for good splitting) by ancestor nodes and save the information on
        # newly discovered constant features to spare computation on descendant
        # nodes.
        while (f_i > n_total_constants and  # Stop early if remaining features
                                            # are constant
                (n_visited_features < max_features or
                 # At least one drawn features must be non constant
                 n_visited_features <= n_found_constants + n_drawn_constants)):

            n_visited_features += 1

            # Loop invariant: elements of features in
            # - [:n_drawn_constant[ holds drawn and known constant features;
            # - [n_drawn_constant:n_known_constant[ holds known constant
            #   features that haven't been drawn yet;
            # - [n_known_constant:n_total_constant[ holds newly found constant
            #   features;
            # - [n_total_constant:f_i[ holds features that haven't been drawn
            #   yet and aren't constant apriori.
            # - [f_i:n_features[ holds features that have been drawn
            #   and aren't constant.

            # Draw a feature at random
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                           random_state)

            if f_j < n_known_constants:
                # f_j in the interval [n_drawn_constants, n_known_constants[
                tmp = features[f_j]
                features[f_j] = features[n_drawn_constants]
                features[n_drawn_constants] = tmp

                n_drawn_constants += 1

            else:
                # f_j in the interval [n_known_constants, f_i - n_found_constants[
                f_j += n_found_constants
                # f_j in the interval [n_total_constants, f_i[

                current.feature = features[f_j]

                # Sort samples along that feature; first copy the feature
                # values for the active samples into Xf, s.t.
                # Xf[i] == X[samples[i], j], so the sort uses the cache more
                # effectively.
                for p in range(start, end):
                    Xf[p] = X[X_sample_stride * samples[p] +
                              X_fx_stride * current.feature]

                sort(Xf + start, samples + start, end - start)

                if Xf[end - 1] <= Xf[start] + FEATURE_THRESHOLD:
                    features[f_j] = features[n_total_constants]
                    features[n_total_constants] = current.feature

                    n_found_constants += 1
                    n_total_constants += 1

                else:
                    f_i -= 1
                    features[f_i], features[f_j] = features[f_j], features[f_i]

                    # Evaluate all splits
                    self.criterion.reset()
                    p = start

                    while p < end:
                        while (p + 1 < end and
                               Xf[p + 1] <= Xf[p] + FEATURE_THRESHOLD):
                            p += 1

                        # (p + 1 >= end) or (X[samples[p + 1], current.feature] >
                        #                    X[samples[p], current.feature])
                        p += 1
                        # (p >= end) or (X[samples[p], current.feature] >
                        #                X[samples[p - 1], current.feature])

                        if p < end:
                            current.pos = p

                            # Reject if min_samples_leaf is not guaranteed
                            if (((current.pos - start) < min_samples_leaf) or
                                    ((end - current.pos) < min_samples_leaf)):
                                continue

                            self.criterion.update(current.pos)

                            # Reject if min_weight_leaf is not satisfied
                            if ((self.criterion.weighted_n_left < min_weight_leaf) or
                                    (self.criterion.weighted_n_right < min_weight_leaf)):
                                continue
                            
                            current.improvement = self.criterion.impurity_improvement(impurity)

                            if current.improvement > best.improvement:
                                self.criterion.children_impurity(&current.impurity_left,
                                                                 &current.impurity_right)
                                current.threshold = (Xf[p - 1] + Xf[p]) / 2.0

                                if current.threshold == Xf[p]:
                                    current.threshold = Xf[p - 1]

                                best = current  # copy

        # Reorganize into samples[start:best.pos] + samples[best.pos:end]
        if best.pos < end:
            partition_end = end
            p = start

            while p < partition_end:
                if X[X_sample_stride * samples[p] +
                     X_fx_stride * best.feature] <= best.threshold:
                    p += 1

                else:
                    partition_end -= 1

                    tmp = samples[partition_end]
                    samples[partition_end] = samples[p]
                    samples[p] = tmp

        # Respect invariant for constant features: the original order of
        # element in features[:n_known_constants] must be preserved for sibling
        # and child nodes
        memcpy(features, constant_features, sizeof(SIZE_t) * n_known_constants)

        # Copy newly found constant features
        memcpy(constant_features + n_known_constants,
               features + n_known_constants,
               sizeof(SIZE_t) * n_found_constants)

        # Return values
        split[0] = best
        n_constant_features[0] = n_total_constants


cdef class UnSupervisedBestSplitter(BestSplitter):
    """Splitter for finding the best split on un-labelled data."""
    cdef UnsupervisedClassificationCriterion criterion_real
    
    def __cinit__(self, UnsupervisedClassificationCriterion criterion, SIZE_t max_features,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  object random_state):
        self.criterion_real = criterion
    
    def __reduce__(self):
        return (UnSupervisedBestSplitter, (self.criterion,
                                           self.max_features,
                                           self.min_samples_leaf,
                                           self.min_weight_leaf,
                                           self.random_state), self.__getstate__())    

    cdef void node_reset(self, SIZE_t start, SIZE_t end,
                         double* weighted_n_node_samples) nogil:
        """Reset splitter on node samples[start:end]."""
        self.start = start
        self.end = end

        self.criterion_real.init2(self.X,
                                  self.X_sample_stride,
                                  self.sample_weight,
                                  self.weighted_n_samples,
                                  self.samples,
                                  start,
                                  end)

        weighted_n_node_samples[0] = self.criterion_real.weighted_n_node_samples
        
        
    cdef void node_split(self, double impurity, SplitRecord* split,
                         SIZE_t* n_constant_features) nogil:
        """Find the best split on node samples[start:end]."""
        # Find the best split
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        cdef SIZE_t* features = self.features
        cdef SIZE_t* constant_features = self.constant_features
        cdef SIZE_t n_features = self.n_features

        cdef DTYPE_t* X = self.X
        cdef DTYPE_t* Xf = self.feature_values
        cdef SIZE_t X_sample_stride = self.X_sample_stride
        cdef SIZE_t X_fx_stride = self.X_fx_stride
        cdef SIZE_t max_features = self.max_features
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef SplitRecord best, current

        cdef SIZE_t f_i = n_features
        cdef SIZE_t f_j, p, tmp
        cdef SIZE_t n_visited_features = 0
        # Number of features discovered to be constant during the split search
        cdef SIZE_t n_found_constants = 0
        # Number of features known to be constant and drawn without replacement
        cdef SIZE_t n_drawn_constants = 0
        cdef SIZE_t n_known_constants = n_constant_features[0]
        # n_total_constants = n_known_constants + n_found_constants
        cdef SIZE_t n_total_constants = n_known_constants
        cdef DTYPE_t current_feature_value
        cdef SIZE_t partition_end

        _init_split(&best, end)

        # Sample up to max_features without replacement using a
        # Fisher-Yates-based algorithm (using the local variables `f_i` and
        # `f_j` to compute a permutation of the `features` array).
        #
        # Skip the CPU intensive evaluation of the impurity criterion for
        # features that were already detected as constant (hence not suitable
        # for good splitting) by ancestor nodes and save the information on
        # newly discovered constant features to spare computation on descendant
        # nodes.
        while (f_i > n_total_constants and  # Stop early if remaining features
                                            # are constant
                (n_visited_features < max_features or
                 # At least one drawn features must be non constant
                 n_visited_features <= n_found_constants + n_drawn_constants)):

            n_visited_features += 1

            # Loop invariant: elements of features in
            # - [:n_drawn_constant[ holds drawn and known constant features;
            # - [n_drawn_constant:n_known_constant[ holds known constant
            #   features that haven't been drawn yet;
            # - [n_known_constant:n_total_constant[ holds newly found constant
            #   features;
            # - [n_total_constant:f_i[ holds features that haven't been drawn
            #   yet and aren't constant apriori.
            # - [f_i:n_features[ holds features that have been drawn
            #   and aren't constant.

            # Draw a feature at random
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                           random_state)

            if f_j < n_known_constants:
                # f_j in the interval [n_drawn_constants, n_known_constants[
                tmp = features[f_j]
                features[f_j] = features[n_drawn_constants]
                features[n_drawn_constants] = tmp

                n_drawn_constants += 1

            else:
                # f_j in the interval [n_known_constants, f_i - n_found_constants[
                f_j += n_found_constants
                # f_j in the interval [n_total_constants, f_i[

                current.feature = features[f_j]

                # Sort samples along that feature; first copy the feature
                # values for the active samples into Xf, s.t.
                # Xf[i] == X[samples[i], j], so the sort uses the cache more
                # effectively.
                for p in range(start, end):
                    Xf[p] = X[X_sample_stride * samples[p] +
                              X_fx_stride * current.feature]

                sort(Xf + start, samples + start, end - start)
                self.criterion_real.sortS()        

                if Xf[end - 1] <= Xf[start] + FEATURE_THRESHOLD:
                    features[f_j] = features[n_total_constants]
                    features[n_total_constants] = current.feature

                    n_found_constants += 1
                    n_total_constants += 1

                else:
                    f_i -= 1
                    features[f_i], features[f_j] = features[f_j], features[f_i]

                    # Evaluate all splits
                    self.criterion.reset()
                    p = start

                    while p < end:
                        while (p + 1 < end and
                               Xf[p + 1] <= Xf[p] + FEATURE_THRESHOLD):
                            p += 1

                        # (p + 1 >= end) or (X[samples[p + 1], current.feature] >
                        #                    X[samples[p], current.feature])
                        p += 1
                        # (p >= end) or (X[samples[p], current.feature] >
                        #                X[samples[p - 1], current.feature])

                        if p < end:
                            current.pos = p

                            # Reject if min_samples_leaf is not guaranteed
                            if (((current.pos - start) < min_samples_leaf) or
                                    ((end - current.pos) < min_samples_leaf)):
                                continue

                            self.criterion.update(current.pos)

                            # Reject if min_weight_leaf is not satisfied
                            if ((self.criterion.weighted_n_left < min_weight_leaf) or
                                    (self.criterion.weighted_n_right < min_weight_leaf)):
                                continue

                            current.improvement = self.criterion.impurity_improvement(impurity)

                            if current.improvement > best.improvement:
                                self.criterion.children_impurity(&current.impurity_left,
                                                                 &current.impurity_right)
                                current.threshold = (Xf[p - 1] + Xf[p]) / 2.0

                                if current.threshold == Xf[p]:
                                    current.threshold = Xf[p - 1]

                                best = current  # copy

        # Reorganize into samples[start:best.pos] + samples[best.pos:end]
        if best.pos < end:
            partition_end = end
            p = start

            while p < partition_end:
                if X[X_sample_stride * samples[p] +
                     X_fx_stride * best.feature] <= best.threshold:
                    p += 1

                else:
                    partition_end -= 1

                    tmp = samples[partition_end]
                    samples[partition_end] = samples[p]
                    samples[p] = tmp

        # Respect invariant for constant features: the original order of
        # element in features[:n_known_constants] must be preserved for sibling
        # and child nodes
        memcpy(features, constant_features, sizeof(SIZE_t) * n_known_constants)

        # Copy newly found constant features
        memcpy(constant_features + n_known_constants,
               features + n_known_constants,
               sizeof(SIZE_t) * n_found_constants)

        # Return values
        #with gil: print 'e-best', best.pos, best.feature, best.threshold
        split[0] = best
        n_constant_features[0] = n_total_constants


# Sort n-element arrays pointed to by Xf and samples, simultaneously,
# by the values in Xf. Algorithm: Introsort (Musser, SP&E, 1997).
cdef inline void sort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    cdef int maxd = 2 * <int>log(n)
    introsort(Xf, samples, n, maxd)


cdef inline void swap(DTYPE_t* Xf, SIZE_t* samples, SIZE_t i, SIZE_t j) nogil:
    # Helper for sort
    Xf[i], Xf[j] = Xf[j], Xf[i]
    samples[i], samples[j] = samples[j], samples[i]


cdef inline DTYPE_t median3(DTYPE_t* Xf, SIZE_t n) nogil:
    # Median of three pivot selection, after Bentley and McIlroy (1993).
    # Engineering a sort function. SP&E. Requires 8/3 comparisons on average.
    cdef DTYPE_t a = Xf[0], b = Xf[n / 2], c = Xf[n - 1]
    if a < b:
        if b < c:
            return b
        elif a < c:
            return c
        else:
            return a
    elif b < c:
        if a < c:
            return a
        else:
            return c
    else:
        return b


# Introsort with median of 3 pivot selection and 3-way partition function
# (robust to repeated elements, e.g. lots of zero features).
cdef void introsort(DTYPE_t* Xf, SIZE_t *samples, SIZE_t n, int maxd) nogil:
    cdef DTYPE_t pivot
    cdef SIZE_t i, l, r

    while n > 1:
        if maxd <= 0:   # max depth limit exceeded ("gone quadratic")
            heapsort(Xf, samples, n)
            return
        maxd -= 1

        pivot = median3(Xf, n)

        # Three-way partition.
        i = l = 0
        r = n
        while i < r:
            if Xf[i] < pivot:
                swap(Xf, samples, i, l)
                i += 1
                l += 1
            elif Xf[i] > pivot:
                r -= 1
                swap(Xf, samples, i, r)
            else:
                i += 1

        introsort(Xf, samples, l, maxd)
        Xf += r
        samples += r
        n -= r


cdef inline void sift_down(DTYPE_t* Xf, SIZE_t* samples,
                           SIZE_t start, SIZE_t end) nogil:
    # Restore heap order in Xf[start:end] by moving the max element to start.
    cdef SIZE_t child, maxind, root

    root = start
    while True:
        child = root * 2 + 1

        # find max of root, left child, right child
        maxind = root
        if child < end and Xf[maxind] < Xf[child]:
            maxind = child
        if child + 1 < end and Xf[maxind] < Xf[child + 1]:
            maxind = child + 1

        if maxind == root:
            break
        else:
            swap(Xf, samples, root, maxind)
            root = maxind


cdef void heapsort(DTYPE_t* Xf, SIZE_t* samples, SIZE_t n) nogil:
    cdef SIZE_t start, end

    # heapify
    start = (n - 2) / 2
    end = n
    while True:
        sift_down(Xf, samples, start, end)
        if start == 0:
            break
        start -= 1

    # sort by shrinking the heap, putting the max element immediately after it
    end = n - 1
    while end > 0:
        swap(Xf, samples, 0, end)
        sift_down(Xf, samples, 0, end)
        end = end - 1


cdef class RandomSplitter(BaseDenseSplitter):
    """Splitter for finding the best random split."""
    def __reduce__(self):
        return (RandomSplitter, (self.criterion,
                                 self.max_features,
                                 self.min_samples_leaf,
                                 self.min_weight_leaf,
                                 self.random_state), self.__getstate__())

    cdef void node_split(self, double impurity, SplitRecord* split,
                         SIZE_t* n_constant_features) nogil:
        """Find the best random split on node samples[start:end]."""
        # Draw random splits and pick the best
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        cdef SIZE_t* features = self.features
        cdef SIZE_t* constant_features = self.constant_features
        cdef SIZE_t n_features = self.n_features

        cdef DTYPE_t* X = self.X
        cdef DTYPE_t* Xf = self.feature_values
        cdef SIZE_t X_sample_stride = self.X_sample_stride
        cdef SIZE_t X_fx_stride = self.X_fx_stride
        cdef SIZE_t max_features = self.max_features
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef SplitRecord best, current

        cdef SIZE_t f_i = n_features
        cdef SIZE_t f_j, p, tmp
        # Number of features discovered to be constant during the split search
        cdef SIZE_t n_found_constants = 0
        # Number of features known to be constant and drawn without replacement
        cdef SIZE_t n_drawn_constants = 0
        cdef SIZE_t n_known_constants = n_constant_features[0]
        # n_total_constants = n_known_constants + n_found_constants
        cdef SIZE_t n_total_constants = n_known_constants
        cdef SIZE_t n_visited_features = 0
        cdef DTYPE_t min_feature_value
        cdef DTYPE_t max_feature_value
        cdef DTYPE_t current_feature_value
        cdef SIZE_t partition_end

        _init_split(&best, end)

        # Sample up to max_features without replacement using a
        # Fisher-Yates-based algorithm (using the local variables `f_i` and
        # `f_j` to compute a permutation of the `features` array).
        #
        # Skip the CPU intensive evaluation of the impurity criterion for
        # features that were already detected as constant (hence not suitable
        # for good splitting) by ancestor nodes and save the information on
        # newly discovered constant features to spare computation on descendant
        # nodes.
        while (f_i > n_total_constants and  # Stop early if remaining features
                                            # are constant
                (n_visited_features < max_features or
                 # At least one drawn features must be non constant
                 n_visited_features <= n_found_constants + n_drawn_constants)):
            n_visited_features += 1

            # Loop invariant: elements of features in
            # - [:n_drawn_constant[ holds drawn and known constant features;
            # - [n_drawn_constant:n_known_constant[ holds known constant
            #   features that haven't been drawn yet;
            # - [n_known_constant:n_total_constant[ holds newly found constant
            #   features;
            # - [n_total_constant:f_i[ holds features that haven't been drawn
            #   yet and aren't constant apriori.
            # - [f_i:n_features[ holds features that have been drawn
            #   and aren't constant.

            # Draw a feature at random
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                           random_state)

            if f_j < n_known_constants:
                # f_j in the interval [n_drawn_constants, n_known_constants[
                tmp = features[f_j]
                features[f_j] = features[n_drawn_constants]
                features[n_drawn_constants] = tmp

                n_drawn_constants += 1

            else:
                # f_j in the interval [n_known_constants, f_i - n_found_constants[
                f_j += n_found_constants
                # f_j in the interval [n_total_constants, f_i[

                current.feature = features[f_j]

                # Find min, max
                min_feature_value = X[X_sample_stride * samples[start] +
                                      X_fx_stride * current.feature]
                max_feature_value = min_feature_value
                Xf[start] = min_feature_value

                for p in range(start + 1, end):
                    current_feature_value = X[X_sample_stride * samples[p] +
                                              X_fx_stride * current.feature]
                    Xf[p] = current_feature_value

                    if current_feature_value < min_feature_value:
                        min_feature_value = current_feature_value
                    elif current_feature_value > max_feature_value:
                        max_feature_value = current_feature_value

                if max_feature_value <= min_feature_value + FEATURE_THRESHOLD:
                    features[f_j] = features[n_total_constants]
                    features[n_total_constants] = current.feature

                    n_found_constants += 1
                    n_total_constants += 1

                else:
                    f_i -= 1
                    features[f_i], features[f_j] = features[f_j], features[f_i]

                    # Draw a random threshold
                    current.threshold = rand_uniform(min_feature_value,
                                                     max_feature_value,
                                                     random_state)

                    if current.threshold == max_feature_value:
                        current.threshold = min_feature_value

                    # Partition
                    partition_end = end
                    p = start
                    while p < partition_end:
                        current_feature_value = Xf[p]
                        if current_feature_value <= current.threshold:
                            p += 1
                        else:
                            partition_end -= 1

                            Xf[p] = Xf[partition_end]
                            Xf[partition_end] = current_feature_value

                            tmp = samples[partition_end]
                            samples[partition_end] = samples[p]
                            samples[p] = tmp

                    current.pos = partition_end

                    # Reject if min_samples_leaf is not guaranteed
                    if (((current.pos - start) < min_samples_leaf) or
                            ((end - current.pos) < min_samples_leaf)):
                        continue

                    # Evaluate split
                    self.criterion.reset()
                    self.criterion.update(current.pos)

                    # Reject if min_weight_leaf is not satisfied
                    if ((self.criterion.weighted_n_left < min_weight_leaf) or
                            (self.criterion.weighted_n_right < min_weight_leaf)):
                        continue

                    current.improvement = self.criterion.impurity_improvement(impurity)

                    if current.improvement > best.improvement:
                        self.criterion.children_impurity(&current.impurity_left,
                                                         &current.impurity_right)
                        best = current  # copy

        # Reorganize into samples[start:best.pos] + samples[best.pos:end]
        if best.pos < end and current.feature != best.feature:
            partition_end = end
            p = start

            while p < partition_end:
                if X[X_sample_stride * samples[p] +
                     X_fx_stride * best.feature] <= best.threshold:
                    p += 1

                else:
                    partition_end -= 1

                    tmp = samples[partition_end]
                    samples[partition_end] = samples[p]
                    samples[p] = tmp

        # Respect invariant for constant features: the original order of
        # element in features[:n_known_constants] must be preserved for sibling
        # and child nodes
        memcpy(features, constant_features, sizeof(SIZE_t) * n_known_constants)

        # Copy newly found constant features
        memcpy(constant_features + n_known_constants,
               features + n_known_constants,
               sizeof(SIZE_t) * n_found_constants)

        # Return values
        split[0] = best
        n_constant_features[0] = n_total_constants


cdef class PresortBestSplitter(BaseDenseSplitter):
    """Splitter for finding the best split, using presorting."""
    cdef DTYPE_t* X_old
    cdef np.ndarray X_argsorted
    cdef INT32_t* X_argsorted_ptr
    cdef SIZE_t X_argsorted_stride

    cdef SIZE_t n_total_samples
    cdef unsigned char* sample_mask

    def __cinit__(self, Criterion criterion, SIZE_t max_features,
                  SIZE_t min_samples_leaf,
                  double min_weight_leaf,
                  object random_state):
        # Initialize pointers
        self.X_old = NULL
        self.X_argsorted_ptr = NULL
        self.X_argsorted_stride = 0
        self.sample_mask = NULL

    def __dealloc__(self):
        """Destructor."""
        free(self.sample_mask)

    def __reduce__(self):
        return (PresortBestSplitter, (self.criterion,
                                      self.max_features,
                                      self.min_samples_leaf,
                                      self.min_weight_leaf,
                                      self.random_state), self.__getstate__())

    cdef void init(self, object X,
                   np.ndarray[DOUBLE_t, ndim=2, mode="c"] y,
                   DOUBLE_t* sample_weight) except *:

        cdef void* sample_mask = NULL

        # Call parent initializer
        BaseDenseSplitter.init(self, X, y, sample_weight)

        cdef np.ndarray X_ndarray = X

        # Pre-sort X
        if self.X_old != self.X:
            self.X_old = self.X
            self.X_argsorted = np.asfortranarray(np.argsort(X_ndarray, axis=0),
                                                 dtype=np.int32)
            self.X_argsorted_ptr = <INT32_t*> self.X_argsorted.data
            self.X_argsorted_stride = (<SIZE_t> self.X_argsorted.strides[1] /
                                       <SIZE_t> self.X_argsorted.itemsize)

            self.n_total_samples = X.shape[0]
            sample_mask = safe_realloc(&self.sample_mask, self.n_total_samples)
            memset(sample_mask, 0, self.n_total_samples)

    cdef void node_split(self, double impurity, SplitRecord* split,
                         SIZE_t* n_constant_features) nogil:
        """Find the best split on node samples[start:end]."""
        # Find the best split
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        cdef SIZE_t* features = self.features
        cdef SIZE_t* constant_features = self.constant_features
        cdef SIZE_t n_features = self.n_features

        cdef DTYPE_t* X = self.X
        cdef DTYPE_t* Xf = self.feature_values
        cdef SIZE_t X_sample_stride = self.X_sample_stride
        cdef SIZE_t X_fx_stride = self.X_fx_stride
        cdef INT32_t* X_argsorted = self.X_argsorted_ptr
        cdef SIZE_t X_argsorted_stride = self.X_argsorted_stride
        cdef SIZE_t n_total_samples = self.n_total_samples
        cdef unsigned char* sample_mask = self.sample_mask

        cdef SIZE_t max_features = self.max_features
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef SplitRecord best, current

        cdef SIZE_t f_i = n_features
        cdef SIZE_t f_j, p
        # Number of features discovered to be constant during the split search
        cdef SIZE_t n_found_constants = 0
        # Number of features known to be constant and drawn without replacement
        cdef SIZE_t n_drawn_constants = 0
        cdef SIZE_t n_known_constants = n_constant_features[0]
        # n_total_constants = n_known_constants + n_found_constants
        cdef SIZE_t n_total_constants = n_known_constants
        cdef SIZE_t n_visited_features = 0
        cdef SIZE_t partition_end
        cdef SIZE_t i, j

        _init_split(&best, end)

        # Set sample mask
        for p in range(start, end):
            sample_mask[samples[p]] = 1

        # Sample up to max_features without replacement using a
        # Fisher-Yates-based algorithm (using the local variables `f_i` and
        # `f_j` to compute a permutation of the `features` array).
        #
        # Skip the CPU intensive evaluation of the impurity criterion for
        # features that were already detected as constant (hence not suitable
        # for good splitting) by ancestor nodes and save the information on
        # newly discovered constant features to spare computation on descendant
        # nodes.
        while (f_i > n_total_constants and  # Stop early if remaining features
                                            # are constant
                (n_visited_features < max_features or
                 # At least one drawn features must be non constant
                 n_visited_features <= n_found_constants + n_drawn_constants)):
            n_visited_features += 1

            # Loop invariant: elements of features in
            # - [:n_drawn_constant[ holds drawn and known constant features;
            # - [n_drawn_constant:n_known_constant[ holds known constant
            #   features that haven't been drawn yet;
            # - [n_known_constant:n_total_constant[ holds newly found constant
            #   features;
            # - [n_total_constant:f_i[ holds features that haven't been drawn
            #   yet and aren't constant apriori.
            # - [f_i:n_features[ holds features that have been drawn
            #   and aren't constant.

            # Draw a feature at random
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                           random_state)

            if f_j < n_known_constants:
                # f_j is in [n_drawn_constants, n_known_constants[
                tmp = features[f_j]
                features[f_j] = features[n_drawn_constants]
                features[n_drawn_constants] = tmp

                n_drawn_constants += 1

            else:
                # f_j in the interval [n_known_constants, f_i - n_found_constants[
                f_j += n_found_constants
                # f_j in the interval [n_total_constants, f_i[

                current.feature = features[f_j]

                # Extract ordering from X_argsorted
                p = start

                for i in range(n_total_samples):
                    j = X_argsorted[X_argsorted_stride * current.feature + i]
                    if sample_mask[j] == 1:
                        samples[p] = j
                        Xf[p] = X[X_sample_stride * j +
                                  X_fx_stride * current.feature]
                        p += 1

                # Evaluate all splits
                if Xf[end - 1] <= Xf[start] + FEATURE_THRESHOLD:
                    features[f_j] = features[n_total_constants]
                    features[n_total_constants] = current.feature

                    n_found_constants += 1
                    n_total_constants += 1

                else:
                    f_i -= 1
                    features[f_i], features[f_j] = features[f_j], features[f_i]

                    self.criterion.reset()
                    p = start

                    while p < end:
                        while (p + 1 < end and
                               Xf[p + 1] <= Xf[p] + FEATURE_THRESHOLD):
                            p += 1

                        # (p + 1 >= end) or (X[samples[p + 1], current.feature] >
                        #                    X[samples[p], current.feature])
                        p += 1
                        # (p >= end) or (X[samples[p], current.feature] >
                        #                X[samples[p - 1], current.feature])

                        if p < end:
                            current.pos = p

                            # Reject if min_samples_leaf is not guaranteed
                            if (((current.pos - start) < min_samples_leaf) or
                                    ((end - current.pos) < min_samples_leaf)):
                                continue

                            self.criterion.update(current.pos)

                            # Reject if min_weight_leaf is not satisfied
                            if ((self.criterion.weighted_n_left < min_weight_leaf) or
                                    (self.criterion.weighted_n_right < min_weight_leaf)):
                                continue

                            current.improvement = self.criterion.impurity_improvement(impurity)

                            if current.improvement > best.improvement:
                                self.criterion.children_impurity(&current.impurity_left,
                                                                 &current.impurity_right)

                                current.threshold = (Xf[p - 1] + Xf[p]) / 2.0
                                if current.threshold == Xf[p]:
                                    current.threshold = Xf[p - 1]

                                best = current  # copy

        # Reorganize into samples[start:best.pos] + samples[best.pos:end]
        if best.pos < end:
            partition_end = end
            p = start

            while p < partition_end:
                if X[X_sample_stride * samples[p] +
                     X_fx_stride * best.feature] <= best.threshold:
                    p += 1

                else:
                    partition_end -= 1

                    tmp = samples[partition_end]
                    samples[partition_end] = samples[p]
                    samples[p] = tmp

        # Reset sample mask
        for p in range(start, end):
            sample_mask[samples[p]] = 0

        # Respect invariant for constant features: the original order of
        # element in features[:n_known_constants] must be preserved for sibling
        # and child nodes
        memcpy(features, constant_features, sizeof(SIZE_t) * n_known_constants)

        # Copy newly found constant features
        memcpy(constant_features + n_known_constants,
               features + n_known_constants,
               sizeof(SIZE_t) * n_found_constants)

        # Return values
        split[0] = best
        n_constant_features[0] = n_total_constants


cdef class BaseSparseSplitter(Splitter):
    # The sparse splitter works only with csc sparse matrix format
    cdef DTYPE_t* X_data
    cdef INT32_t* X_indices
    cdef INT32_t* X_indptr

    cdef SIZE_t n_total_samples

    cdef SIZE_t* index_to_samples
    cdef SIZE_t* sorted_samples

    def __cinit__(self, Criterion criterion, SIZE_t max_features,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  object random_state):
        # Parent __cinit__ is automatically called

        self.X_data = NULL
        self.X_indices = NULL
        self.X_indptr = NULL

        self.n_total_samples = 0

        self.index_to_samples = NULL
        self.sorted_samples = NULL

    def __dealloc__(self):
        """Deallocate memory"""
        free(self.index_to_samples)
        free(self.sorted_samples)

    cdef void init(self,
                   object X,
                   np.ndarray[DOUBLE_t, ndim=2, mode="c"] y,
                   DOUBLE_t* sample_weight) except *:
        """Initialize the splitter."""

        # Call parent init
        Splitter.init(self, X, y, sample_weight)

        if not isinstance(X, csc_matrix):
            raise ValueError("X should be in csc format")

        cdef SIZE_t* samples = self.samples
        cdef SIZE_t n_samples = self.n_samples

        # Initialize X
        cdef np.ndarray[dtype=DTYPE_t, ndim=1] data = X.data
        cdef np.ndarray[dtype=INT32_t, ndim=1] indices = X.indices
        cdef np.ndarray[dtype=INT32_t, ndim=1] indptr = X.indptr
        cdef SIZE_t n_total_samples = X.shape[0]

        self.X_data = <DTYPE_t*> data.data
        self.X_indices = <INT32_t*> indices.data
        self.X_indptr = <INT32_t*> indptr.data
        self.n_total_samples = n_total_samples

        # Initialize auxiliary array used to perform split
        safe_realloc(&self.index_to_samples, n_total_samples * sizeof(SIZE_t))
        safe_realloc(&self.sorted_samples, n_samples * sizeof(SIZE_t))

        cdef SIZE_t* index_to_samples = self.index_to_samples
        cdef SIZE_t p
        for p in range(n_total_samples):
            index_to_samples[p] = -1

        for p in range(n_samples):
            index_to_samples[samples[p]] = p

    cdef inline SIZE_t _partition(self, double threshold,
                                  SIZE_t end_negative, SIZE_t start_positive,
                                  SIZE_t zero_pos) nogil:
        """Partition samples[start:end] based on threshold"""

        cdef double value
        cdef SIZE_t partition_end
        cdef SIZE_t p

        cdef DTYPE_t* Xf = self.feature_values
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t* index_to_samples = self.index_to_samples

        if threshold < 0.:
            p = self.start
            partition_end = end_negative
        elif threshold > 0.:
            p = start_positive
            partition_end = self.end
        else:
            # Data are already split
            return zero_pos

        while p < partition_end:
            value = Xf[p]

            if value <= threshold:
                p += 1

            else:
                partition_end -= 1

                Xf[p] = Xf[partition_end]
                Xf[partition_end] = value
                sparse_swap(index_to_samples, samples, p, partition_end)

        return partition_end

    cdef inline void extract_nnz(self, SIZE_t feature,
                                 SIZE_t* end_negative, SIZE_t* start_positive,
                                 bint* is_samples_sorted) nogil:
        """Extract and partition values for a given feature

        The extracted values are partitioned between negative values
        Xf[start:end_negative[0]] and positive values Xf[start_positive[0]:end].
        The samples and index_to_samples are modified according to this
        partition.

        The extraction corresponds to the intersection between the arrays
        X_indices[indptr_start:indptr_end] and samples[start:end].
        This is done efficiently using either an index_to_samples based approach
        or binary search based approach.

        Parameters
        ----------
        feature : SIZE_t,
            Index of the feature we want to extract non zero value.


        end_negative, start_positive : SIZE_t*, SIZE_t*,
            Return extracted non zero values in self.samples[start:end] where
            negative values are in self.feature_values[start:end_negative[0]]
            and positive values are in
            self.feature_values[start_positive[0]:end].

        is_samples_sorted : bint*,
            If is_samples_sorted, then self.sorted_samples[start:end] will be
            the sorted version of self.samples[start:end].

        """
        cdef SIZE_t indptr_start = self.X_indptr[feature],
        cdef SIZE_t indptr_end = self.X_indptr[feature + 1]
        cdef SIZE_t n_indices = <SIZE_t>(indptr_end - indptr_start)
        cdef SIZE_t n_samples = self.end - self.start

        # Use binary search if n_samples * log(n_indices) <
        # n_indices and index_to_samples approach otherwise.
        # O(n_samples * log(n_indices)) is the running time of binary
        # search and O(n_indices) is the running time of index_to_samples
        # approach.
        if ((1 - is_samples_sorted[0]) * n_samples * log(n_samples) +
                n_samples * log(n_indices) < EXTRACT_NNZ_SWITCH * n_indices):
            extract_nnz_binary_search(self.X_indices, self.X_data,
                                      indptr_start, indptr_end,
                                      self.samples, self.start, self.end,
                                      self.index_to_samples,
                                      self.feature_values,
                                      end_negative, start_positive,
                                      self.sorted_samples, is_samples_sorted)

        # Using an index to samples  technique to extract non zero values
        # index_to_samples is a mapping from X_indices to samples
        else:
            extract_nnz_index_to_samples(self.X_indices, self.X_data,
                                         indptr_start, indptr_end,
                                         self.samples, self.start, self.end,
                                         self.index_to_samples,
                                         self.feature_values,
                                         end_negative, start_positive)


cdef int compare_SIZE_t(const void* a, const void* b) nogil:
    """Comparison function for sort"""
    return <int>((<SIZE_t*>a)[0] - (<SIZE_t*>b)[0])


cdef inline void binary_search(INT32_t* sorted_array,
                               INT32_t start, INT32_t end,
                               SIZE_t value, SIZE_t* index,
                               INT32_t* new_start) nogil:
    """Return the index of value in the sorted array

    If not found, return -1. new_start is the last pivot + 1
    """
    cdef INT32_t pivot
    index[0] = -1
    while start < end:
        pivot = start + (end - start) / 2

        if sorted_array[pivot] == value:
            index[0] = pivot
            start = pivot + 1
            break

        if sorted_array[pivot] < value:
            start = pivot + 1
        else:
            end = pivot
    new_start[0] = start


cdef inline void extract_nnz_index_to_samples(INT32_t* X_indices,
                                              DTYPE_t* X_data,
                                              INT32_t indptr_start,
                                              INT32_t indptr_end,
                                              SIZE_t* samples,
                                              SIZE_t start,
                                              SIZE_t end,
                                              SIZE_t* index_to_samples,
                                              DTYPE_t* Xf,
                                              SIZE_t* end_negative,
                                              SIZE_t* start_positive) nogil:
    """Extract and partition values for a feature using index_to_samples

    Complexity is O(indptr_end - indptr_start).
    """
    cdef INT32_t k
    cdef SIZE_t index
    cdef SIZE_t end_negative_ = start
    cdef SIZE_t start_positive_ = end

    for k in range(indptr_start, indptr_end):
        if start <= index_to_samples[X_indices[k]] < end:
            if X_data[k] > 0:
                start_positive_ -= 1
                Xf[start_positive_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, start_positive_)


            elif X_data[k] < 0:
                Xf[end_negative_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, end_negative_)
                end_negative_ += 1

    # Returned values
    end_negative[0] = end_negative_
    start_positive[0] = start_positive_


cdef inline void extract_nnz_binary_search(INT32_t* X_indices,
                                           DTYPE_t* X_data,
                                           INT32_t indptr_start,
                                           INT32_t indptr_end,
                                           SIZE_t* samples,
                                           SIZE_t start,
                                           SIZE_t end,
                                           SIZE_t* index_to_samples,
                                           DTYPE_t* Xf,
                                           SIZE_t* end_negative,
                                           SIZE_t* start_positive,
                                           SIZE_t* sorted_samples,
                                           bint* is_samples_sorted) nogil:
    """Extract and partition values for a given feature using binary search

    If n_samples = end - start and n_indices = indptr_end - indptr_start,
    the complexity is

        O((1 - is_samples_sorted[0]) * n_samples * log(n_samples) +
          n_samples * log(n_indices)).
    """
    cdef SIZE_t n_samples

    if not is_samples_sorted[0]:
        n_samples = end - start
        memcpy(sorted_samples + start, samples + start,
               n_samples * sizeof(SIZE_t))
        qsort(sorted_samples + start, n_samples, sizeof(SIZE_t),
              compare_SIZE_t)
        is_samples_sorted[0] = 1

    while (indptr_start < indptr_end and
           sorted_samples[start] > X_indices[indptr_start]):
        indptr_start += 1

    while (indptr_start < indptr_end and
           sorted_samples[end - 1] < X_indices[indptr_end - 1]):
        indptr_end -= 1

    cdef SIZE_t p = start
    cdef SIZE_t index
    cdef SIZE_t k
    cdef SIZE_t end_negative_ = start
    cdef SIZE_t start_positive_ = end

    while (p < end and indptr_start < indptr_end):
        # Find index of sorted_samples[p] in X_indices
        binary_search(X_indices, indptr_start, indptr_end,
                      sorted_samples[p], &k, &indptr_start)

        if k != -1:
             # If k != -1, we have found a non zero value

            if X_data[k] > 0:
                start_positive_ -= 1
                Xf[start_positive_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, start_positive_)


            elif X_data[k] < 0:
                Xf[end_negative_] = X_data[k]
                index = index_to_samples[X_indices[k]]
                sparse_swap(index_to_samples, samples, index, end_negative_)
                end_negative_ += 1
        p += 1

    # Returned values
    end_negative[0] = end_negative_
    start_positive[0] = start_positive_


cdef inline void sparse_swap(SIZE_t* index_to_samples, SIZE_t* samples,
                             SIZE_t pos_1, SIZE_t pos_2) nogil  :
    """Swap sample pos_1 and pos_2 preserving sparse invariant"""
    samples[pos_1], samples[pos_2] =  samples[pos_2], samples[pos_1]
    index_to_samples[samples[pos_1]] = pos_1
    index_to_samples[samples[pos_2]] = pos_2


cdef class BestSparseSplitter(BaseSparseSplitter):
    """Splitter for finding the best split, using the sparse data."""

    def __reduce__(self):
        return (BestSparseSplitter, (self.criterion,
                                     self.max_features,
                                     self.min_samples_leaf,
                                     self.min_weight_leaf,
                                     self.random_state), self.__getstate__())

    cdef void node_split(self, double impurity, SplitRecord* split,
                         SIZE_t* n_constant_features) nogil:
        """Find the best split on node samples[start:end], using sparse
           features.
        """
        # Find the best split
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        cdef INT32_t* X_indices = self.X_indices
        cdef INT32_t* X_indptr = self.X_indptr
        cdef DTYPE_t* X_data = self.X_data

        cdef SIZE_t* features = self.features
        cdef SIZE_t* constant_features = self.constant_features
        cdef SIZE_t n_features = self.n_features

        cdef DTYPE_t* Xf = self.feature_values
        cdef SIZE_t* sorted_samples = self.sorted_samples
        cdef SIZE_t* index_to_samples = self.index_to_samples
        cdef SIZE_t max_features = self.max_features
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef SplitRecord best, current
        _init_split(&best, end)

        cdef SIZE_t f_i = n_features
        cdef SIZE_t f_j, p, tmp
        cdef SIZE_t n_visited_features = 0
        # Number of features discovered to be constant during the split search
        cdef SIZE_t n_found_constants = 0
        # Number of features known to be constant and drawn without replacement
        cdef SIZE_t n_drawn_constants = 0
        cdef SIZE_t n_known_constants = n_constant_features[0]
        # n_total_constants = n_known_constants + n_found_constants
        cdef SIZE_t n_total_constants = n_known_constants
        cdef DTYPE_t current_feature_value

        cdef SIZE_t p_next
        cdef SIZE_t p_prev
        cdef bint is_samples_sorted = 0  # indicate is sorted_samples is
                                         # inititialized

        # We assume implicitely that end_positive = end and
        # start_negative = start
        cdef SIZE_t start_positive
        cdef SIZE_t end_negative

        # Sample up to max_features without replacement using a
        # Fisher-Yates-based algorithm (using the local variables `f_i` and
        # `f_j` to compute a permutation of the `features` array).
        #
        # Skip the CPU intensive evaluation of the impurity criterion for
        # features that were already detected as constant (hence not suitable
        # for good splitting) by ancestor nodes and save the information on
        # newly discovered constant features to spare computation on descendant
        # nodes.
        while (f_i > n_total_constants and  # Stop early if remaining features
                                            # are constant
                (n_visited_features < max_features or
                 # At least one drawn features must be non constant
                 n_visited_features <= n_found_constants + n_drawn_constants)):

            n_visited_features += 1

            # Loop invariant: elements of features in
            # - [:n_drawn_constant[ holds drawn and known constant features;
            # - [n_drawn_constant:n_known_constant[ holds known constant
            #   features that haven't been drawn yet;
            # - [n_known_constant:n_total_constant[ holds newly found constant
            #   features;
            # - [n_total_constant:f_i[ holds features that haven't been drawn
            #   yet and aren't constant apriori.
            # - [f_i:n_features[ holds features that have been drawn
            #   and aren't constant.

            # Draw a feature at random
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                           random_state)

            if f_j < n_known_constants:
                # f_j in the interval [n_drawn_constants, n_known_constants[
                tmp = features[f_j]
                features[f_j] = features[n_drawn_constants]
                features[n_drawn_constants] = tmp

                n_drawn_constants += 1

            else:
                # f_j in the interval [n_known_constants, f_i - n_found_constants[
                f_j += n_found_constants
                # f_j in the interval [n_total_constants, f_i[

                current.feature = features[f_j]
                self.extract_nnz(current.feature,
                                 &end_negative, &start_positive,
                                 &is_samples_sorted)

                # Sort the positive and negative parts of `Xf`
                sort(Xf + start, samples + start, end_negative - start)
                sort(Xf + start_positive, samples + start_positive,
                     end - start_positive)

                # Update index_to_samples to take into account the sort
                for p in range(start, end_negative):
                    index_to_samples[samples[p]] = p
                for p in range(start_positive, end):
                    index_to_samples[samples[p]] = p

                # Add one or two zeros in Xf, if there is any
                if end_negative < start_positive:
                    start_positive -= 1
                    Xf[start_positive] = 0.

                    if end_negative != start_positive:
                        Xf[end_negative] = 0.
                        end_negative += 1

                if Xf[end - 1] <= Xf[start] + FEATURE_THRESHOLD:
                    features[f_j] = features[n_total_constants]
                    features[n_total_constants] = current.feature

                    n_found_constants += 1
                    n_total_constants += 1

                else:
                    f_i -= 1
                    features[f_i], features[f_j] = features[f_j], features[f_i]

                    # Evaluate all splits
                    self.criterion.reset()
                    p = start

                    while p < end:
                        if p + 1 != end_negative:
                            p_next = p + 1
                        else:
                            p_next = start_positive

                        while (p_next < end and
                               Xf[p_next] <= Xf[p] + FEATURE_THRESHOLD):
                            p = p_next
                            if p + 1 != end_negative:
                                p_next = p + 1
                            else:
                                p_next = start_positive


                        # (p_next >= end) or (X[samples[p_next], current.feature] >
                        #                     X[samples[p], current.feature])
                        p_prev = p
                        p = p_next
                        # (p >= end) or (X[samples[p], current.feature] >
                        #                X[samples[p_prev], current.feature])


                        if p < end:
                            current.pos = p

                            # Reject if min_samples_leaf is not guaranteed
                            if (((current.pos - start) < min_samples_leaf) or
                                    ((end - current.pos) < min_samples_leaf)):
                                continue

                            self.criterion.update(current.pos)

                            # Reject if min_weight_leaf is not satisfied
                            if ((self.criterion.weighted_n_left < min_weight_leaf) or
                                    (self.criterion.weighted_n_right < min_weight_leaf)):
                                continue

                            current.improvement = self.criterion.impurity_improvement(impurity)
                            if current.improvement > best.improvement:
                                self.criterion.children_impurity(&current.impurity_left,
                                                                 &current.impurity_right)

                                current.threshold = (Xf[p_prev] + Xf[p]) / 2.0
                                if current.threshold == Xf[p]:
                                    current.threshold = Xf[p_prev]

                                best = current

        # Reorganize into samples[start:best.pos] + samples[best.pos:end]
        if best.pos < end:
            self.extract_nnz(best.feature, &end_negative, &start_positive,
                             &is_samples_sorted)

            self._partition(best.threshold, end_negative, start_positive,
                            best.pos)

        # Respect invariant for constant features: the original order of
        # element in features[:n_known_constants] must be preserved for sibling
        # and child nodes
        memcpy(features, constant_features, sizeof(SIZE_t) * n_known_constants)

        # Copy newly found constant features
        memcpy(constant_features + n_known_constants,
               features + n_known_constants,
               sizeof(SIZE_t) * n_found_constants)

        # Return values
        split[0] = best      
        n_constant_features[0] = n_total_constants


cdef class RandomSparseSplitter(BaseSparseSplitter):
    """Splitter for finding a random split, using the sparse data."""

    def __reduce__(self):
        return (RandomSparseSplitter, (self.criterion,
                                       self.max_features,
                                       self.min_samples_leaf,
                                       self.min_weight_leaf,
                                       self.random_state), self.__getstate__())

    cdef void node_split(self, double impurity, SplitRecord* split,
                         SIZE_t* n_constant_features) nogil:
        """Find a random split on node samples[start:end], using sparse
           features.
        """
        # Find the best split
        cdef SIZE_t* samples = self.samples
        cdef SIZE_t start = self.start
        cdef SIZE_t end = self.end

        cdef INT32_t* X_indices = self.X_indices
        cdef INT32_t* X_indptr = self.X_indptr
        cdef DTYPE_t* X_data = self.X_data

        cdef SIZE_t* features = self.features
        cdef SIZE_t* constant_features = self.constant_features
        cdef SIZE_t n_features = self.n_features

        cdef DTYPE_t* Xf = self.feature_values
        cdef SIZE_t* sorted_samples = self.sorted_samples
        cdef SIZE_t* index_to_samples = self.index_to_samples
        cdef SIZE_t max_features = self.max_features
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef UINT32_t* random_state = &self.rand_r_state

        cdef SplitRecord best, current
        _init_split(&best, end)

        cdef DTYPE_t current_feature_value

        cdef SIZE_t f_i = n_features
        cdef SIZE_t f_j, p, tmp
        cdef SIZE_t n_visited_features = 0
        # Number of features discovered to be constant during the split search
        cdef SIZE_t n_found_constants = 0
        # Number of features known to be constant and drawn without replacement
        cdef SIZE_t n_drawn_constants = 0
        cdef SIZE_t n_known_constants = n_constant_features[0]
        # n_total_constants = n_known_constants + n_found_constants
        cdef SIZE_t n_total_constants = n_known_constants
        cdef SIZE_t partition_end

        cdef DTYPE_t min_feature_value
        cdef DTYPE_t max_feature_value

        cdef bint is_samples_sorted = 0  # indicate that sorted_samples is
                                         # inititialized

        # We assume implicitely that end_positive = end and
        # start_negative = start
        cdef SIZE_t start_positive
        cdef SIZE_t end_negative

        # Sample up to max_features without replacement using a
        # Fisher-Yates-based algorithm (using the local variables `f_i` and
        # `f_j` to compute a permutation of the `features` array).
        #
        # Skip the CPU intensive evaluation of the impurity criterion for
        # features that were already detected as constant (hence not suitable
        # for good splitting) by ancestor nodes and save the information on
        # newly discovered constant features to spare computation on descendant
        # nodes.
        while (f_i > n_total_constants and  # Stop early if remaining features
                                            # are constant
                (n_visited_features < max_features or
                 # At least one drawn features must be non constant
                 n_visited_features <= n_found_constants + n_drawn_constants)):

            n_visited_features += 1

            # Loop invariant: elements of features in
            # - [:n_drawn_constant[ holds drawn and known constant features;
            # - [n_drawn_constant:n_known_constant[ holds known constant
            #   features that haven't been drawn yet;
            # - [n_known_constant:n_total_constant[ holds newly found constant
            #   features;
            # - [n_total_constant:f_i[ holds features that haven't been drawn
            #   yet and aren't constant apriori.
            # - [f_i:n_features[ holds features that have been drawn
            #   and aren't constant.

            # Draw a feature at random
            f_j = rand_int(n_drawn_constants, f_i - n_found_constants,
                           random_state)

            if f_j < n_known_constants:
                # f_j in the interval [n_drawn_constants, n_known_constants[
                tmp = features[f_j]
                features[f_j] = features[n_drawn_constants]
                features[n_drawn_constants] = tmp

                n_drawn_constants += 1

            else:
                # f_j in the interval [n_known_constants, f_i - n_found_constants[
                f_j += n_found_constants
                # f_j in the interval [n_total_constants, f_i[

                current.feature = features[f_j]

                self.extract_nnz(current.feature,
                                 &end_negative, &start_positive,
                                 &is_samples_sorted)

                # Add one or two zeros in Xf, if there is any
                if end_negative < start_positive:
                    start_positive -= 1
                    Xf[start_positive] = 0.

                    if end_negative != start_positive:
                        Xf[end_negative] = 0.
                        end_negative += 1

                # Find min, max in Xf[start:end_negative]
                min_feature_value = Xf[start]
                max_feature_value = min_feature_value

                for p in range(start, end_negative):
                    current_feature_value = Xf[p]

                    if current_feature_value < min_feature_value:
                        min_feature_value = current_feature_value
                    elif current_feature_value > max_feature_value:
                        max_feature_value = current_feature_value

                # Update min, max given Xf[start_positive:end]
                for p in range(start_positive, end):
                    current_feature_value = Xf[p]

                    if current_feature_value < min_feature_value:
                        min_feature_value = current_feature_value
                    elif current_feature_value > max_feature_value:
                        max_feature_value = current_feature_value

                if max_feature_value <= min_feature_value + FEATURE_THRESHOLD:
                    features[f_j] = features[n_total_constants]
                    features[n_total_constants] = current.feature

                    n_found_constants += 1
                    n_total_constants += 1

                else:
                    f_i -= 1
                    features[f_i], features[f_j] = features[f_j], features[f_i]

                    # Draw a random threshold
                    current.threshold = rand_uniform(min_feature_value,
                                                     max_feature_value,
                                                     random_state)

                    if current.threshold == max_feature_value:
                        current.threshold = min_feature_value

                    # Partition
                    current.pos = self._partition(current.threshold,
                                                  end_negative,
                                                  start_positive,
                                                  start_positive +
                                                  (Xf[start_positive] == 0.))

                    # Reject if min_samples_leaf is not guaranteed
                    if (((current.pos - start) < min_samples_leaf) or
                            ((end - current.pos) < min_samples_leaf)):
                        continue

                    # Evaluate split
                    self.criterion.reset()
                    self.criterion.update(current.pos)

                    # Reject if min_weight_leaf is not satisfied
                    if ((self.criterion.weighted_n_left < min_weight_leaf) or
                            (self.criterion.weighted_n_right < min_weight_leaf)):
                        continue

                    current.improvement = self.criterion.impurity_improvement(impurity)

                    if current.improvement > best.improvement:
                        self.criterion.children_impurity(&current.impurity_left,
                                                         &current.impurity_right)
                        best = current

        # Reorganize into samples[start:best.pos] + samples[best.pos:end]
        if best.pos < end and current.feature != best.feature:
            self.extract_nnz(best.feature, &end_negative, &start_positive,
                             &is_samples_sorted)

            self._partition(best.threshold, end_negative, start_positive,
                            best.pos)

        # Respect invariant for constant features: the original order of
        # element in features[:n_known_constants] must be preserved for sibling
        # and child nodes
        memcpy(features, constant_features, sizeof(SIZE_t) * n_known_constants)

        # Copy newly found constant features
        memcpy(constant_features + n_known_constants,
               features + n_known_constants,
               sizeof(SIZE_t) * n_found_constants)

        # Return values
        split[0] = best
        n_constant_features[0] = n_total_constants


# =============================================================================
# Tree builders
# =============================================================================
cdef class TreeBuilder:
    """Interface for different tree building strategies. """

    cpdef build(self, Tree tree, object X, np.ndarray y,
                np.ndarray sample_weight=None):
        """Build a decision tree from the training set (X, y)."""
        pass

    cdef inline _check_input(self, object X, np.ndarray y,
                             np.ndarray sample_weight):
        """Check input dtype, layout and format"""
        if issparse(X):
            X = X.tocsc()
            X.sort_indices()

            if X.data.dtype != DTYPE:
                X.data = np.ascontiguousarray(X.data, dtype=DTYPE)

            if X.indices.dtype != np.int32 or X.indptr.dtype != np.int32:
                raise ValueError("No support for np.int64 index based "
                                 "sparse matrices")

        elif X.dtype != DTYPE:
            # since we have to copy we will make it fortran for efficiency
            X = np.asfortranarray(X, dtype=DTYPE)

        if y.dtype != DOUBLE or not y.flags.contiguous:
            y = np.ascontiguousarray(y, dtype=DOUBLE)

        if (sample_weight is not None and
            (sample_weight.dtype != DOUBLE or
            not sample_weight.flags.contiguous)):
                sample_weight = np.asarray(sample_weight, dtype=DOUBLE,
                                           order="C")

        return X, y, sample_weight

# Depth first builder ---------------------------------------------------------

cdef class DepthFirstTreeBuilder(TreeBuilder):
    """Build a decision tree in depth-first fashion."""

    def __cinit__(self, Splitter splitter, SIZE_t min_samples_split,
                  SIZE_t min_samples_leaf, double min_weight_leaf,
                  SIZE_t max_depth):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth

    cpdef build(self, Tree tree, object X, np.ndarray y,
                np.ndarray sample_weight=None):
        """Build a decision tree from the training set (X, y)."""

        # check input
        X, y, sample_weight = self._check_input(X, y, sample_weight)

        cdef DOUBLE_t* sample_weight_ptr = NULL
        if sample_weight is not None:
            sample_weight_ptr = <DOUBLE_t*> sample_weight.data

        # Initial capacity
        cdef int init_capacity

        if tree.max_depth <= 10:
            init_capacity = (2 ** (tree.max_depth + 1)) - 1
        else:
            init_capacity = 2047

        tree._resize(init_capacity)

        # Parameters
        cdef Splitter splitter = self.splitter
        cdef SIZE_t max_depth = self.max_depth
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef SIZE_t min_samples_split = self.min_samples_split

        # Recursive partition (without actual recursion)
        splitter.init(X, y, sample_weight_ptr)

        cdef SIZE_t start
        cdef SIZE_t end
        cdef SIZE_t depth
        cdef SIZE_t parent
        cdef bint is_left
        cdef SIZE_t n_node_samples = splitter.n_samples
        cdef double weighted_n_samples = splitter.weighted_n_samples
        cdef double weighted_n_node_samples
        cdef SplitRecord split
        cdef SIZE_t node_id

        cdef double threshold
        cdef double impurity = INFINITY
        cdef SIZE_t n_constant_features
        cdef bint is_leaf
        cdef bint first = 1
        cdef SIZE_t max_depth_seen = -1
        cdef int rc = 0

        cdef Stack stack = Stack(INITIAL_STACK_SIZE)
        cdef StackRecord stack_record

        # push root node onto stack
        rc = stack.push(0, n_node_samples, 0, _TREE_UNDEFINED, 0, INFINITY, 0)
        if rc == -1:
            # got return code -1 - out-of-memory
            raise MemoryError()

        with nogil:
            while not stack.is_empty():
                stack.pop(&stack_record)

                start = stack_record.start
                end = stack_record.end
                depth = stack_record.depth
                parent = stack_record.parent
                is_left = stack_record.is_left
                impurity = stack_record.impurity
                n_constant_features = stack_record.n_constant_features

                n_node_samples = end - start
                splitter.node_reset(start, end, &weighted_n_node_samples)

                is_leaf = ((depth >= max_depth) or
                           (n_node_samples < min_samples_split) or
                           (n_node_samples < 2 * min_samples_leaf) or
                           (weighted_n_node_samples < min_weight_leaf))

                if first:
                    impurity = splitter.node_impurity()
                    first = 0

                is_leaf = is_leaf or (impurity <= MIN_IMPURITY_SPLIT)

                if not is_leaf:
                    splitter.node_split(impurity, &split, &n_constant_features)
                    is_leaf = is_leaf or (split.pos >= end)

                node_id = tree._add_node(parent, is_left, is_leaf, split.feature,
                                         split.threshold, impurity, n_node_samples,
                                         weighted_n_node_samples)

                if is_leaf:
                    # Don't store value for internal nodes
                    splitter.node_value(tree.value +
                                        node_id * tree.value_stride)

                else:
                    # Push right child on stack
                    rc = stack.push(split.pos, end, depth + 1, node_id, 0,
                                    split.impurity_right, n_constant_features)
                    if rc == -1:
                        break

                    # Push left child on stack
                    rc = stack.push(start, split.pos, depth + 1, node_id, 1,
                                    split.impurity_left, n_constant_features)
                    if rc == -1:
                        break

                if depth > max_depth_seen:
                    max_depth_seen = depth

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

            if rc >= 0:
                tree.max_depth = max_depth_seen
        if rc == -1:
            raise MemoryError()


# Best first builder ----------------------------------------------------------

cdef inline int _add_to_frontier(PriorityHeapRecord* rec,
                                 PriorityHeap frontier) nogil:
    """Adds record ``rec`` to the priority queue ``frontier``; returns -1
    on memory-error. """
    return frontier.push(rec.node_id, rec.start, rec.end, rec.pos, rec.depth,
                         rec.is_leaf, rec.improvement, rec.impurity,
                         rec.impurity_left, rec.impurity_right)


cdef class BestFirstTreeBuilder(TreeBuilder):
    """Build a decision tree in best-first fashion.

    The best node to expand is given by the node at the frontier that has the
    highest impurity improvement.

    NOTE: this TreeBuilder will ignore ``tree.max_depth`` .
    """
    cdef SIZE_t max_leaf_nodes

    def __cinit__(self, Splitter splitter, SIZE_t min_samples_split,
                  SIZE_t min_samples_leaf,  min_weight_leaf,
                  SIZE_t max_depth, SIZE_t max_leaf_nodes):
        self.splitter = splitter
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_leaf = min_weight_leaf
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes

    cpdef build(self, Tree tree, object X, np.ndarray y,
                np.ndarray sample_weight=None):
        """Build a decision tree from the training set (X, y)."""

        # check input
        X, y, sample_weight = self._check_input(X, y, sample_weight)

        cdef DOUBLE_t* sample_weight_ptr = NULL
        if sample_weight is not None:
            sample_weight_ptr = <DOUBLE_t*> sample_weight.data

        # Parameters
        cdef Splitter splitter = self.splitter
        cdef SIZE_t max_leaf_nodes = self.max_leaf_nodes
        cdef SIZE_t min_samples_leaf = self.min_samples_leaf
        cdef double min_weight_leaf = self.min_weight_leaf
        cdef SIZE_t min_samples_split = self.min_samples_split

        # Recursive partition (without actual recursion)
        splitter.init(X, y, sample_weight_ptr)

        cdef PriorityHeap frontier = PriorityHeap(INITIAL_STACK_SIZE)
        cdef PriorityHeapRecord record
        cdef PriorityHeapRecord split_node_left
        cdef PriorityHeapRecord split_node_right

        cdef SIZE_t n_node_samples = splitter.n_samples
        cdef SIZE_t max_split_nodes = max_leaf_nodes - 1
        cdef bint is_leaf
        cdef SIZE_t max_depth_seen = -1
        cdef int rc = 0
        cdef Node* node

        # Initial capacity
        cdef SIZE_t init_capacity = max_split_nodes + max_leaf_nodes
        tree._resize(init_capacity)

        with nogil:
            # add root to frontier
            rc = self._add_split_node(splitter, tree, 0, n_node_samples,
                                      INFINITY, IS_FIRST, IS_LEFT, NULL, 0,
                                      &split_node_left)
            if rc >= 0:
                rc = _add_to_frontier(&split_node_left, frontier)
        if rc == -1:
            raise MemoryError()

        with nogil:
            while not frontier.is_empty():
                frontier.pop(&record)

                node = &tree.nodes[record.node_id]
                is_leaf = (record.is_leaf or max_split_nodes <= 0)

                if is_leaf:
                    # Node is not expandable; set node as leaf
                    node.left_child = _TREE_LEAF
                    node.right_child = _TREE_LEAF
                    node.feature = _TREE_UNDEFINED
                    node.threshold = _TREE_UNDEFINED

                else:
                    # Node is expandable

                    # Decrement number of split nodes available
                    max_split_nodes -= 1

                    # Compute left split node
                    rc = self._add_split_node(splitter, tree,
                                              record.start, record.pos,
                                              record.impurity_left,
                                              IS_NOT_FIRST, IS_LEFT, node,
                                              record.depth + 1,
                                              &split_node_left)
                    if rc == -1:
                        break

                    # tree.nodes may have changed
                    node = &tree.nodes[record.node_id]

                    # Compute right split node
                    rc = self._add_split_node(splitter, tree, record.pos,
                                              record.end,
                                              record.impurity_right,
                                              IS_NOT_FIRST, IS_NOT_LEFT, node,
                                              record.depth + 1,
                                              &split_node_right)
                    if rc == -1:
                        break

                    # Add nodes to queue
                    rc = _add_to_frontier(&split_node_left, frontier)
                    if rc == -1:
                        break

                    rc = _add_to_frontier(&split_node_right, frontier)
                    if rc == -1:
                        break

                if record.depth > max_depth_seen:
                    max_depth_seen = record.depth

            if rc >= 0:
                rc = tree._resize_c(tree.node_count)

            if rc >= 0:
                tree.max_depth = max_depth_seen

        if rc == -1:
            raise MemoryError()

    cdef inline int _add_split_node(self, Splitter splitter, Tree tree,
                                    SIZE_t start, SIZE_t end, double impurity,
                                    bint is_first, bint is_left, Node* parent,
                                    SIZE_t depth,
                                    PriorityHeapRecord* res) nogil:
        """Adds node w/ partition ``[start, end)`` to the frontier. """
        cdef SplitRecord split
        cdef SIZE_t node_id
        cdef SIZE_t n_node_samples
        cdef SIZE_t n_constant_features = 0
        cdef double weighted_n_samples = splitter.weighted_n_samples
        cdef double weighted_n_node_samples
        cdef bint is_leaf
        cdef SIZE_t n_left, n_right
        cdef double imp_diff

        splitter.node_reset(start, end, &weighted_n_node_samples)

        if is_first:
            impurity = splitter.node_impurity()

        n_node_samples = end - start
        is_leaf = ((depth > self.max_depth) or
                   (n_node_samples < self.min_samples_split) or
                   (n_node_samples < 2 * self.min_samples_leaf) or
                   (weighted_n_node_samples < self.min_weight_leaf) or
                   (impurity <= MIN_IMPURITY_SPLIT))

        if not is_leaf:
            splitter.node_split(impurity, &split, &n_constant_features)
            is_leaf = is_leaf or (split.pos >= end)

        node_id = tree._add_node(parent - tree.nodes
                                 if parent != NULL
                                 else _TREE_UNDEFINED,
                                 is_left, is_leaf,
                                 split.feature, split.threshold, impurity, n_node_samples,
                                 weighted_n_node_samples)
        if node_id == <SIZE_t>(-1):
            return -1

        # compute values also for split nodes (might become leafs later).
        splitter.node_value(tree.value + node_id * tree.value_stride)

        res.node_id = node_id
        res.start = start
        res.end = end
        res.depth = depth
        res.impurity = impurity

        if not is_leaf:
            # is split node
            res.pos = split.pos
            res.is_leaf = 0
            res.improvement = split.improvement
            res.impurity_left = split.impurity_left
            res.impurity_right = split.impurity_right

        else:
            # is leaf => 0 improvement
            res.pos = end
            res.is_leaf = 1
            res.improvement = 0.0
            res.impurity_left = impurity
            res.impurity_right = impurity

        return 0


# =============================================================================
# Tree
# =============================================================================

cdef class Tree:
    """Array-based representation of a binary decision tree.

    The binary tree is represented as a number of parallel arrays. The i-th
    element of each array holds information about the node `i`. Node 0 is the
    tree's root. You can find a detailed description of all arrays in
    `_tree.pxd`. NOTE: Some of the arrays only apply to either leaves or split
    nodes, resp. In this case the values of nodes of the other type are
    arbitrary!

    Attributes
    ----------
    node_count : int
        The number of nodes (internal nodes + leaves) in the tree.

    capacity : int
        The current capacity (i.e., size) of the arrays, which is at least as
        great as `node_count`.

    max_depth : int
        The maximal depth of the tree.

    children_left : array of int, shape [node_count]
        children_left[i] holds the node id of the left child of node i.
        For leaves, children_left[i] == TREE_LEAF. Otherwise,
        children_left[i] > i. This child handles the case where
        X[:, feature[i]] <= threshold[i].

    children_right : array of int, shape [node_count]
        children_right[i] holds the node id of the right child of node i.
        For leaves, children_right[i] == TREE_LEAF. Otherwise,
        children_right[i] > i. This child handles the case where
        X[:, feature[i]] > threshold[i].

    feature : array of int, shape [node_count]
        feature[i] holds the feature to split on, for the internal node i.

    threshold : array of double, shape [node_count]
        threshold[i] holds the threshold for the internal node i.

    value : array of double, shape [node_count, n_outputs, max_n_classes]
        Contains the constant prediction value of each node.

    impurity : array of double, shape [node_count]
        impurity[i] holds the impurity (i.e., the value of the splitting
        criterion) at node i.

    n_node_samples : array of int, shape [node_count]
        n_node_samples[i] holds the number of training samples reaching node i.

    weighted_n_node_samples : array of int, shape [node_count]
        weighted_n_node_samples[i] holds the weighted number of training samples
        reaching node i.
    """
    # Wrap for outside world.
    # WARNING: these reference the current `nodes` and `value` buffers, which
    # must not be be freed by a subsequent memory allocation.
    # (i.e. through `_resize` or `__setstate__`)
    property n_classes:
        def __get__(self):
            # it's small; copy for memory safety
            return sizet_ptr_to_ndarray(self.n_classes, self.n_outputs).copy()

    property children_left:
        def __get__(self):
            return self._get_node_ndarray()['left_child'][:self.node_count]

    property children_right:
        def __get__(self):
            return self._get_node_ndarray()['right_child'][:self.node_count]

    property feature:
        def __get__(self):
            return self._get_node_ndarray()['feature'][:self.node_count]

    property threshold:
        def __get__(self):
            return self._get_node_ndarray()['threshold'][:self.node_count]

    property impurity:
        def __get__(self):
            return self._get_node_ndarray()['impurity'][:self.node_count]

    property n_node_samples:
        def __get__(self):
            return self._get_node_ndarray()['n_node_samples'][:self.node_count]

    property weighted_n_node_samples:
        def __get__(self):
            return self._get_node_ndarray()['weighted_n_node_samples'][:self.node_count]

    property value:
        def __get__(self):
            return self._get_value_ndarray()[:self.node_count]

    def __cinit__(self, int n_features, np.ndarray[SIZE_t, ndim=1] n_classes,
                  int n_outputs):
        """Constructor."""
        # Input/Output layout
        self.n_features = n_features
        self.n_outputs = n_outputs
        self.n_classes = NULL
        safe_realloc(&self.n_classes, n_outputs)

        self.max_n_classes = np.max(n_classes)
        self.value_stride = n_outputs * self.max_n_classes

        cdef SIZE_t k
        for k in range(n_outputs):
            self.n_classes[k] = n_classes[k]

        # Inner structures
        self.max_depth = 0
        self.node_count = 0
        self.capacity = 0
        self.value = NULL
        self.nodes = NULL

    def __dealloc__(self):
        """Destructor."""
        # Free all inner structures
        free(self.n_classes)
        free(self.value)
        free(self.nodes)

    def __reduce__(self):
        """Reduce re-implementation, for pickling."""
        return (Tree, (self.n_features,
                       sizet_ptr_to_ndarray(self.n_classes, self.n_outputs),
                       self.n_outputs), self.__getstate__())

    def __getstate__(self):
        """Getstate re-implementation, for pickling."""
        d = {}
        d["node_count"] = self.node_count
        d["nodes"] = self._get_node_ndarray()
        d["values"] = self._get_value_ndarray()
        return d

    def __setstate__(self, d):
        """Setstate re-implementation, for unpickling."""
        self.node_count = d["node_count"]

        if 'nodes' not in d:
            raise ValueError('You have loaded Tree version which '
                             'cannot be imported')

        node_ndarray = d['nodes']
        value_ndarray = d['values']

        value_shape = (node_ndarray.shape[0], self.n_outputs,
                       self.max_n_classes)
        if (node_ndarray.ndim != 1 or
                node_ndarray.dtype != NODE_DTYPE or
                not node_ndarray.flags.c_contiguous or
                value_ndarray.shape != value_shape or
                not value_ndarray.flags.c_contiguous or
                value_ndarray.dtype != np.float64):
            raise ValueError('Did not recognise loaded array layout')

        self.capacity = node_ndarray.shape[0]
        if self._resize_c(self.capacity) != 0:
            raise MemoryError("resizing tree to %d" % self.capacity)
        nodes = memcpy(self.nodes, (<np.ndarray> node_ndarray).data,
                       self.capacity * sizeof(Node))
        value = memcpy(self.value, (<np.ndarray> value_ndarray).data,
                       self.capacity * self.value_stride * sizeof(double))

    cdef void _resize(self, SIZE_t capacity) except *:
        """Resize all inner arrays to `capacity`, if `capacity` == -1, then
           double the size of the inner arrays."""
        if self._resize_c(capacity) != 0:
            raise MemoryError()

    # XXX using (size_t)(-1) is ugly, but SIZE_MAX is not available in C89
    # (i.e., older MSVC).
    cdef int _resize_c(self, SIZE_t capacity=<SIZE_t>(-1)) nogil:
        """Guts of _resize. Returns 0 for success, -1 for error."""
        if capacity == self.capacity and self.nodes != NULL:
            return 0

        if capacity == <SIZE_t>(-1):
            if self.capacity == 0:
                capacity = 3  # default initial value
            else:
                capacity = 2 * self.capacity

        # XXX no safe_realloc here because we need to grab the GIL
        cdef void* ptr = realloc(self.nodes, capacity * sizeof(Node))
        if ptr == NULL:
            return -1
        self.nodes = <Node*> ptr
        ptr = realloc(self.value,
                      capacity * self.value_stride * sizeof(double))
        if ptr == NULL:
            return -1
        self.value = <double*> ptr

        # value memory is initialised to 0 to enable classifier argmax
        if capacity > self.capacity:
            memset(<void*>(self.value + self.capacity * self.value_stride), 0,
                   (capacity - self.capacity) * self.value_stride *
                   sizeof(double))

        # if capacity smaller than node_count, adjust the counter
        if capacity < self.node_count:
            self.node_count = capacity

        self.capacity = capacity
        return 0

    cdef SIZE_t _add_node(self, SIZE_t parent, bint is_left, bint is_leaf,
                          SIZE_t feature, double threshold, double impurity,
                          SIZE_t n_node_samples, double weighted_n_node_samples) nogil:
        """Add a node to the tree.

        The new node registers itself as the child of its parent.

        Returns (size_t)(-1) on error.
        """
        cdef SIZE_t node_id = self.node_count

        if node_id >= self.capacity:
            if self._resize_c() != 0:
                return <SIZE_t>(-1)

        cdef Node* node = &self.nodes[node_id]
        node.impurity = impurity
        node.n_node_samples = n_node_samples
        node.weighted_n_node_samples = weighted_n_node_samples

        if parent != _TREE_UNDEFINED:
            if is_left:
                self.nodes[parent].left_child = node_id
            else:
                self.nodes[parent].right_child = node_id

        if is_leaf:
            node.left_child = _TREE_LEAF
            node.right_child = _TREE_LEAF
            node.feature = _TREE_UNDEFINED
            node.threshold = _TREE_UNDEFINED

        else:
            # left_child and right_child will be set later
            node.feature = feature
            node.threshold = threshold

        self.node_count += 1

        return node_id

    cpdef np.ndarray predict(self, object X):
        """Predict target for X."""
        out = self._get_value_ndarray().take(self.apply(X), axis=0,
                                             mode='clip')
        if self.n_outputs == 1:
            out = out.reshape(X.shape[0], self.max_n_classes)
        return out

    cpdef np.ndarray apply(self, object X):
        """Finds the terminal region (=leaf node) for each sample in X."""
        if issparse(X):
            return self._apply_sparse_csr(X)
        else:
            return self._apply_dense(X)


    cdef inline np.ndarray _apply_dense(self, object X):
        """Finds the terminal region (=leaf node) for each sample in X."""

        # Check input
        if not isinstance(X, np.ndarray):
            raise ValueError("X should be in np.ndarray format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef np.ndarray X_ndarray = X
        cdef DTYPE_t* X_ptr = <DTYPE_t*> X_ndarray.data
        cdef SIZE_t X_sample_stride = <SIZE_t> X.strides[0] / <SIZE_t> X.itemsize
        cdef SIZE_t X_fx_stride = <SIZE_t> X.strides[1] / <SIZE_t> X.itemsize
        cdef SIZE_t n_samples = X.shape[0]

        # Initialize output
        cdef np.ndarray[SIZE_t] out = np.zeros((n_samples,), dtype=np.intp)
        cdef SIZE_t* out_ptr = <SIZE_t*> out.data

        # Initialize auxiliary data-structure
        cdef Node* node = NULL
        cdef SIZE_t i = 0

        with nogil:
            for i in range(n_samples):
                node = self.nodes
                # While node not a leaf
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    if X_ptr[X_sample_stride * i +
                             X_fx_stride * node.feature] <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                out_ptr[i] = <SIZE_t>(node - self.nodes)  # node offset

        return out

    cdef inline np.ndarray _apply_sparse_csr(self, object X):
        """Finds the terminal region (=leaf node) for each sample in sparse X.

        """
        # Check input
        if not isinstance(X, csr_matrix):
            raise ValueError("X should be in csr_matrix format, got %s"
                             % type(X))

        if X.dtype != DTYPE:
            raise ValueError("X.dtype should be np.float32, got %s" % X.dtype)

        # Extract input
        cdef np.ndarray[ndim=1, dtype=DTYPE_t] X_data_ndarray = X.data
        cdef np.ndarray[ndim=1, dtype=INT32_t] X_indices_ndarray  = X.indices
        cdef np.ndarray[ndim=1, dtype=INT32_t] X_indptr_ndarray  = X.indptr

        cdef DTYPE_t* X_data = <DTYPE_t*>X_data_ndarray.data
        cdef INT32_t* X_indices = <INT32_t*>X_indices_ndarray.data
        cdef INT32_t* X_indptr = <INT32_t*>X_indptr_ndarray.data

        cdef SIZE_t n_samples = X.shape[0]
        cdef SIZE_t n_features = X.shape[1]

        # Initialize output
        cdef np.ndarray[SIZE_t, ndim=1] out = np.zeros((n_samples,),
                                                       dtype=np.intp)
        cdef SIZE_t* out_ptr = <SIZE_t*> out.data

        # Initialize auxiliary data-structure
        cdef DTYPE_t feature_value = 0.
        cdef Node* node = NULL
        cdef DTYPE_t* X_sample = NULL
        cdef SIZE_t i = 0
        cdef INT32_t k = 0

        # feature_to_sample as a data structure records the last seen sample
        # for each feature; functionally, it is an efficient way to identify
        # which features are nonzero in the present sample.
        cdef SIZE_t* feature_to_sample = NULL

        safe_realloc(&X_sample, n_features * sizeof(DTYPE_t))
        safe_realloc(&feature_to_sample, n_features * sizeof(SIZE_t))

        with nogil:
            memset(feature_to_sample, -1, n_features * sizeof(SIZE_t))

            for i in range(n_samples):
                node = self.nodes

                for k in range(X_indptr[i], X_indptr[i + 1]):
                    feature_to_sample[X_indices[k]] = i
                    X_sample[X_indices[k]] = X_data[k]

                # While node not a leaf
                while node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    if feature_to_sample[node.feature] == i:
                        feature_value = X_sample[node.feature]

                    else:
                        feature_value = 0.

                    if feature_value <= node.threshold:
                        node = &self.nodes[node.left_child]
                    else:
                        node = &self.nodes[node.right_child]

                out_ptr[i] = <SIZE_t>(node - self.nodes)  # node offset

            # Free auxiliary arrays
            free(X_sample)
            free(feature_to_sample)

        return out

    cpdef compute_feature_importances(self, normalize=True):
        """Computes the importance of each feature (aka variable)."""
        cdef Node* left
        cdef Node* right
        cdef Node* nodes = self.nodes
        cdef Node* node = nodes
        cdef Node* end_node = node + self.node_count

        cdef double normalizer = 0.

        cdef np.ndarray[np.float64_t, ndim=1] importances
        importances = np.zeros((self.n_features,))
        cdef DOUBLE_t* importance_data = <DOUBLE_t*>importances.data

        with nogil:
            while node != end_node:
                if node.left_child != _TREE_LEAF:
                    # ... and node.right_child != _TREE_LEAF:
                    left = &nodes[node.left_child]
                    right = &nodes[node.right_child]

                    importance_data[node.feature] += (
                        node.weighted_n_node_samples * node.impurity -
                        left.weighted_n_node_samples * left.impurity -
                        right.weighted_n_node_samples * right.impurity)
                node += 1

        importances /= nodes[0].weighted_n_node_samples

        if normalize:
            normalizer = np.sum(importances)

            if normalizer > 0.0:
                # Avoid dividing by zero (e.g., when root is pure)
                importances /= normalizer

        return importances

    cdef np.ndarray _get_value_ndarray(self):
        """Wraps value as a 3-d NumPy array

        The array keeps a reference to this Tree, which manages the underlying
        memory.
        """
        cdef np.npy_intp shape[3]
        shape[0] = <np.npy_intp> self.node_count
        shape[1] = <np.npy_intp> self.n_outputs
        shape[2] = <np.npy_intp> self.max_n_classes
        cdef np.ndarray arr
        arr = np.PyArray_SimpleNewFromData(3, shape, np.NPY_DOUBLE, self.value)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr

    cdef np.ndarray _get_node_ndarray(self):
        """Wraps nodes as a NumPy struct array

        The array keeps a reference to this Tree, which manages the underlying
        memory. Individual fields are publicly accessible as properties of the
        Tree.
        """
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> self.node_count
        cdef np.npy_intp strides[1]
        strides[0] = sizeof(Node)
        cdef np.ndarray arr
        Py_INCREF(NODE_DTYPE)
        arr = PyArray_NewFromDescr(np.ndarray, <np.dtype> NODE_DTYPE, 1, shape,
                                   strides, <void*> self.nodes,
                                   np.NPY_DEFAULT, None)
        Py_INCREF(self)
        arr.base = <PyObject*> self
        return arr

# =============================================================================
# Utils
# =============================================================================

# safe_realloc(&p, n) resizes the allocation of p to n * sizeof(*p) bytes or
# raises a MemoryError. It never calls free, since that's __dealloc__'s job.
#   cdef DTYPE_t *p = NULL
#   safe_realloc(&p, n)
# is equivalent to p = malloc(n * sizeof(*p)) with error checking.
ctypedef fused realloc_ptr:
    # Add pointer types here as needed.
    (DTYPE_t*)
    (SIZE_t*)
    (unsigned char*)

cdef realloc_ptr safe_realloc(realloc_ptr* p, size_t nelems) except *:
    # sizeof(realloc_ptr[0]) would be more like idiomatic C, but causes Cython
    # 0.20.1 to crash.
    cdef size_t nbytes = nelems * sizeof(p[0][0])
    if nbytes / sizeof(p[0][0]) != nelems:
        # Overflow in the multiplication
        raise MemoryError("could not allocate (%d * %d) bytes"
                          % (nelems, sizeof(p[0][0])))
    cdef realloc_ptr tmp = <realloc_ptr>realloc(p[0], nbytes)
    if tmp == NULL:
        raise MemoryError("could not allocate %d bytes" % nbytes)

    p[0] = tmp
    return tmp  # for convenience


def _realloc_test():
    # Helper for tests. Tries to allocate <size_t>(-1) / 2 * sizeof(size_t)
    # bytes, which will always overflow.
    cdef SIZE_t* p = NULL
    safe_realloc(&p, <size_t>(-1) / 2)
    if p != NULL:
        free(p)
        assert False


# rand_r replacement using a 32bit XorShift generator
# See http://www.jstatsoft.org/v08/i14/paper for details
cdef inline UINT32_t our_rand_r(UINT32_t* seed) nogil:
    seed[0] ^= <UINT32_t>(seed[0] << 13)
    seed[0] ^= <UINT32_t>(seed[0] >> 17)
    seed[0] ^= <UINT32_t>(seed[0] << 5)

    return seed[0] % (<UINT32_t>RAND_R_MAX + 1)

cdef inline np.ndarray sizet_ptr_to_ndarray(SIZE_t* data, SIZE_t size):
    """Encapsulate data into a 1D numpy array of intp's."""
    cdef np.npy_intp shape[1]
    shape[0] = <np.npy_intp> size
    return np.PyArray_SimpleNewFromData(1, shape, np.NPY_INTP, data)

cdef inline SIZE_t rand_int(SIZE_t low, SIZE_t high,
                            UINT32_t* random_state) nogil:
    """Generate a random integer in [0; end)."""
    return low + our_rand_r(random_state) % (high - low)

cdef inline double rand_uniform(double low, double high,
                                UINT32_t* random_state) nogil:
    """Generate a random double in [low; high)."""
    return ((high - low) * <double> our_rand_r(random_state) /
            <double> RAND_R_MAX) + low

cdef inline double log(double x) nogil:
    return ln(x) / ln(2.0)


