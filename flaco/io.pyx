cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc
from flaco cimport includes as lib

np.import_array()


cpdef int read_sql(str stmt, Engine engine):
    cdef bytes stmt_bytes = stmt.encode("utf-8")
    cdef lib.RowIteratorPtr row_iterator
    cdef lib.RowPtr row
    cdef lib.RowColumnNamesArrayPtr row_col_names
    cdef lib.RowTypesArrayPtr row_types
    cdef lib.RowDataArrayPtr row_data

    row_iterator = lib.read_sql(<char*>stmt_bytes, engine.client_ptr)

    # Read first row
    row = lib.next_row(row_iterator)

    # get column names and types
    row_types = lib.row_types(row)
    row_col_names = lib.row_column_names(row)

    # Begin looping until no rows are returned
    while True:
        if row == NULL:
            break
        else:
            row_data = lib.row_data(row, row_types)
            lib.drop(row)

        row = lib.next_row(row_iterator)

        #if data.tag == Data_Tag.Int64:
        #    return data.int64._0
        #else:
        #    return 0
    lib.drop(row_iterator)
    lib.drop(row)
    lib.drop(row_types)
    lib.drop(row_col_names)
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
        self.client_ptr = lib.create_engine(<char*>self.uri)

    def __dealloc__(self):
        if &self.client_ptr != NULL:
            lib.drop(self.client_ptr)
