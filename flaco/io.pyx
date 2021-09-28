
cimport numpy as np
import numpy as np

from flaco.includes cimport read_sql

np.import_array()

ctypedef fused data:
    str
    int


cpdef int read():
    cdef int result
    result = read_sql()
    return result

cdef resize(np.ndarray array, int len):
    array.resize(len)

cdef np.ndarray array_init(int len, np.dtype dtype):
    return np.empty(len, dtype=dtype)
