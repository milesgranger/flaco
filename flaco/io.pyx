cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc
from flaco cimport includes as lib

np.import_array()


cpdef tuple read_sql(str stmt, Engine engine):
    cdef bytes stmt_bytes = stmt.encode("utf-8")

    cdef lib.RowIteratorPtr row_iterator = lib.read_sql(
        <char*>stmt_bytes, engine.client_ptr
    )

    # Read first row
    cdef lib.RowPtr row_ptr = lib.next_row(row_iterator)

    # get column names and row len
    cdef lib.RowColumnNamesArrayPtr row_col_names = lib.row_column_names(row_ptr)
    cdef np.uint32_t n_columns = lib.n_columns(row_ptr)

    # build columns
    cdef np.ndarray columns = np.zeros(shape=n_columns, dtype=object)

    cdef int i
    for i in range(0, n_columns):
        columns[i] = row_col_names[i].decode()

    # np.ndarray would force all internal arrays to be object dtypes
    # b/c each array has a different dtype.
    cdef list output = []

    # Begin looping until no rows are returned
    cdef int row_idx = 0
    cdef lib.RowDataArrayPtr row_data_ptr
    cdef lib.Data data
    while True:
        if row_ptr == NULL:
            break
        else:

            # Insert new row
            row_data_ptr = lib.row_data(row_ptr)
            if row_data_ptr == NULL:
                raise TypeError(f"Unable to pull row data, likely an unsupported type. Check stderr output.")

            if row_idx == 0:
                # Initialize arrays for output
                for i in range(0, n_columns):
                    data = lib.index_row(row_data_ptr, i)
                    output.append(array_init(data, 10))

            # grow arrays if next insert is passed current len
            if output[0].shape[0] < row_idx:
                for i in range(0, n_columns):
                    resize(output[i], output[0].shape[0] + 1000)

            for i in range(0, n_columns):
                data = lib.index_row(row_data_ptr, i)
                insert_data_into_array(data, output[i], row_idx)

            lib.free_row_data_array(row_data_ptr)
            lib.free_row(row_ptr)
            row_idx += 1

        row_ptr = lib.next_row(row_iterator)

    # Ensure arrays are correct size
    if output[0].shape[0] != row_idx:
        for i in range(0, n_columns):
            resize(output[i], row_idx)

    lib.free_row_iter(row_iterator)
    lib.free_row_column_names(row_col_names)

    return columns, output

cdef resize(np.ndarray array, int len):
    array.resize(len, refcheck=False)

cdef np.ndarray array_init(lib.Data data, int len):
    cdef np.ndarray array
    if data.tag == lib.Data_Tag.Int32:
        array = np.empty(shape=len, dtype=object)
    elif data.tag == lib.Data_Tag.Int64:
        array = np.empty(shape=len, dtype=object)
    elif data.tag == lib.Data_Tag.Float32:
        array = np.empty(shape=len, dtype=np.float32)
    elif data.tag == lib.Data_Tag.Float64:
        array = np.empty(shape=len, dtype=np.float64)
    elif data.tag == lib.Data_Tag.String:
        array = np.empty(shape=len, dtype=object)
    else:
        raise ValueError(f"Unsupported tag: {data.tag}")
    return array

cdef void insert_data_into_array(lib.Data data, np.ndarray arr, int idx):

    if data.tag == lib.Data_Tag.Int64:
        arr[idx] = data.int64._0

    elif data.tag == lib.Data_Tag.Int32:
        arr[idx] = data.int32._0

    elif data.tag == lib.Data_Tag.Float64:
        arr[idx] = data.float64._0

    elif data.tag == lib.Data_Tag.Float32:
        arr[idx] = data.float32._0

    elif data.tag == lib.Data_Tag.String:
        if data.string._0 == NULL:
            arr[idx] = None
        else:
            arr[idx] = data.string._0.decode()

    elif data.tag == lib.Data_Tag.Null:
        arr[idx] = None

    else:
        raise ValueError(f"Unsupported Data enum {data.tag}")


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
