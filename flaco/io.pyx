cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc
from flaco.includes cimport read_sql as _read_sql, Data, Data_Tag, drop, create_engine, RowIteratorPtr, next_row

np.import_array()


cpdef int read_sql(str stmt, Engine engine):
    cdef RowIteratorPtr result
    cdef bytes stmt_bytes = stmt.encode("utf-8")
    cdef np.uint32_t* data
    result = _read_sql(<char*>stmt_bytes, engine.client_ptr)

    data = next_row(result)
    data = next_row(result)
    #if data.tag == Data_Tag.Int64:
    #    return data.int64._0
    #else:
    #    return 0
    drop(result)
    return 1

cdef resize(np.ndarray array, int len):
    array.resize(len)

cdef np.ndarray array_init(int len, np.dtype dtype):
    return np.empty(len, dtype=dtype)

cdef class Engine:

    cdef np.uint32_t* client_ptr
    cdef bytes uri

    def __init__(self, str uri):
        self.uri = uri.encode("utf-8")
        self._create_engine()

    cdef _create_engine(self):
        self.client_ptr = <np.uint32_t*>malloc(sizeof(np.uint32_t))
        self.client_ptr = create_engine(<char*>self.uri)

    def __dealloc__(self):
        if &self.client_ptr != NULL:
            drop(self.client_ptr)
