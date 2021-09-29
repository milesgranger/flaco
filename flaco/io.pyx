
cimport numpy as np
import numpy as np

from flaco.includes cimport read_sql as _read_sql, Data, Data_Tag

np.import_array()


cpdef int read_sql():
    cdef Data result
    result = _read_sql()

    if result.tag == Data_Tag.Int64:
        return result.int64._0
    else:
        return 0

cdef resize(np.ndarray array, int len):
    array.resize(len)

cdef np.ndarray array_init(int len, np.dtype dtype):
    return np.empty(len, dtype=dtype)
