cimport numpy as np
import numpy as np
from numpy import random
import cython
from libc.math cimport exp
from libc.stdlib cimport malloc, free

@cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
def calc_school_rankings(
    list households,
    list schools,
    double[:] alpha,
    double[:] optimal_fraction,
    double[:] utility_at_max,
    long[:] categories,
    double[:, :] distance_utilities,
    double[:] household_utilities,
):
    cdef int n_households = len(households)
    cdef int n_schools = len(schools)
    cdef int i
    cdef int j

    cdef double[:, :] compositions = np.zeros((n_schools, 2), dtype=np.float64)
    cdef int[:] households_indices = np.zeros(n_households, dtype=np.intc)
    for i in range(n_schools):
        compositions[i, 0] = schools[i].composition[0]
        compositions[i, 1] = schools[i].composition[1]
    for j in range(n_households):
        households_indices[j] = households[j].idx
    np.asarray(households_indices).sort()

    optimal_fraction = np.take(optimal_fraction, households_indices)
    utility_at_max = np.take(utility_at_max, households_indices)
    alpha = np.take(alpha, households_indices)
    distance_utilities = np.take(distance_utilities, households_indices, axis=0)

    cdef double[:, :] utilities = np.take(compositions, categories, axis=1)

    # Compute utilities
    for i in range(n_schools):
        for j in range(n_households):
            if utilities[i, j] <= optimal_fraction[j]:
                utilities[i, j] = utilities[i, j] / optimal_fraction[j]
            else:
                utilities[i, j] = utility_at_max[j] + (1 - utilities[i, j]) * (1 - utility_at_max[j]) / (
                    1 - optimal_fraction[j]
                )

            # Combine composition utilities with distance utilities
            utilities[i, j] = utilities[i, j] * alpha[j] + distance_utilities[j, i] * (1 - alpha[j])

    # Compute school ranking for each student
    cdef int k
    cdef double exp_sum
    cdef list ranking
    cdef list students
    cdef int[:] ranked_indices = np.zeros(n_schools, dtype=np.intc)
    for j in range(n_households):
        exp_sum = 0
        for i in range(n_schools):
            utilities[i, j] = exp(<double>50.0 * (utilities[i, j] - household_utilities[j]))
            exp_sum = exp_sum + utilities[i, j]
        for i in range(n_schools):
            utilities[i, j] = utilities[i, j] / exp_sum
            ranked_indices[i] = i

        argsort(utilities[:, j], ranked_indices)
        ranked_indices = ranked_indices[::-1]

        ranking = []
        for i in range(n_schools):
            ranking.append(schools[ranked_indices[i]])
        
        students = households[j].students
        n_students = len(students)
        for k in range(n_students):
            students[k].school_preference = ranking


cdef extern from "stdlib.h":
    ctypedef void const_void "const void"
    void qsort(void *base, int nmemb, int size,
            int(*compar)(const_void *, const_void *)) nogil


cdef struct IndexedElement:
    np.ulong_t index
    np.float64_t value


cdef int _compare(const_void *a, const_void *b):
    cdef np.float64_t v = (<IndexedElement*> a).value-(<IndexedElement*> b).value
    if v < 0: return -1
    if v >= 0: return 1


cpdef argsort(double[:] data, int[:] order):
    cdef np.ulong_t i
    cdef np.ulong_t n = data.shape[0]
    
    # Allocate index tracking array.
    cdef IndexedElement *order_struct = <IndexedElement *> malloc(n * sizeof(IndexedElement))
    
    # Copy data into index tracking array.
    for i in range(n):
        order_struct[i].index = i
        order_struct[i].value = data[i]
        
    # Sort index tracking array.
    qsort(<void *> order_struct, n, sizeof(IndexedElement), _compare)
    
    # Copy indices from index tracking array to output array.
    for i in range(n):
        order[i] = order_struct[i].index
        
    # Free index tracking array.
    free(order_struct)