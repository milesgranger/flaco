cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free
from cython cimport view
from flaco cimport includes as lib

np.import_array()


cpdef tuple read_sql(str stmt, Engine engine):
    cdef bytes stmt_bytes = stmt.encode("utf-8")
    cdef lib.RowIteratorPtr row_iterator
    cdef lib.RowPtr row
    cdef lib.RowColumnNamesArrayPtr row_col_names
    cdef lib.RowTypesArrayPtr row_types
    cdef lib.RowDataArrayPtr row_data_ptr
    cdef lib.Data data
    cdef np.uint32_t n_columns

    row_iterator = lib.read_sql(<char*>stmt_bytes, engine.client_ptr)

    # Read first row
    row = lib.next_row(row_iterator)

    # get column names and types
    row_types = lib.row_types(row)
    row_col_names = lib.row_column_names(row)
    n_columns = lib.n_columns(row)

    # build columns
    columns = np.zeros(shape=n_columns, dtype=object)
    cdef int i
    for i in range(0, n_columns):
        print(row_col_names[i])
        columns[i] = row_col_names[i].decode()
    print(f"Done, columns: {columns}")

    # build arrays based on types
    for i in range(0, n_columns):
        print(row_types[i])

    # np.ndarray would force all internal arrays to be object dtypes
    # b/c each array has a different dtype.
    cdef list output = []

    # Begin looping until no rows are returned
    cdef int row_idx = 0
    while True:
        if row == NULL:
            break
        else:

            # Insert new row
            row_data_ptr = lib.row_data(row, row_types)

            if row_idx == 0:
                # Initialize arrays for output
                for i in range(0, n_columns):
                    data = lib.index_row(row_data_ptr, i)
                    output.append(array_init(data, 10))

            # grow arrays if next insert is passed current len
            if output[0].shape[0] < row_idx:
                for i in range(0, n_columns):
                    resize(output[i], output[0].shape[0] + 100)

            for i in range(0, n_columns):
                data = lib.index_row(row_data_ptr, i)
                if data.tag == lib.Data_Tag.Int64:
                    output[i][row_idx] = data.int64._0
                elif data.tag == lib.Data_Tag.Int32:
                    output[i][row_idx] = data.int32._0
                elif data.tag == lib.Data_Tag.Float64:
                    output[i][row_idx] = data.float64._0
                elif data.tag == lib.Data_Tag.Float32:
                    output[i][row_idx] = data.float32._0
                elif data.tag == lib.Data_Tag.String:
                    output[i][row_idx] = data.string._0
            row_idx += 1

        row = lib.next_row(row_iterator)

    # Ensure arrays are correct size
    if output[0].shape[0] != row_idx:
        for i in range(0, n_columns):
            resize(output[i], row_idx)

    return columns, output

cdef resize(np.ndarray array, int len):
    array.resize(len, refcheck=False)

cdef np.ndarray array_init(lib.Data data, int len):
    cdef np.ndarray array
    if data.tag == lib.Data_Tag.Int32:
        array = np.empty(shape=len, dtype=np.int32)
    elif data.tag == lib.Data_Tag.Int64:
        array = np.empty(shape=len, dtype=np.int64)
    elif data.tag == lib.Data_Tag.Float32:
        array = np.empty(shape=len, dtype=np.float32)
    elif data.tag == lib.Data_Tag.Float64:
        array = np.empty(shape=len, dtype=np.float64)
    elif data.tag == lib.Data_Tag.String:
        array = np.empty(shape=len, dtype=object)
    else:
        raise ValueError(f"Unsupported tag: {data.tag}")
    return array

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
