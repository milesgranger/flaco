# distutils: language=c
# cython: language_level=3

cimport numpy as np
import numpy as np
from libc.stdlib cimport malloc, free
from flaco cimport includes as lib


np.import_array()

cdef extern from "numpy/arrayobject.h":
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)


cpdef dict read_sql(str stmt, Database db, int n_rows=-1):
    cdef bytes stmt_bytes = stmt.encode("utf-8")
    cdef np.int32_t _n_rows = n_rows
    cdef char *exc = <char*>malloc(sizeof(char))
    cdef str err_msg

    cdef lib.RowIteratorPtr row_iterator = lib.read_sql(
        <char*>stmt_bytes, db.db_ptr, &exc
    )
    if exc != NULL:
        err_msg = exc.decode()
        free(exc)
        raise ValueError(err_msg)

    # Read first row
    cdef lib.RowPtr row_ptr = lib.next_row(row_iterator)

    if row_ptr == NULL:
        lib.free_row_iter(row_iterator)
        return dict()  # query returned no rows

    cdef lib.RowDataArrayPtr row_data_array_ptr = lib.init_row_data_array(row_ptr)

    # get column names and row len
    cdef lib.RowColumnNamesArrayPtr row_col_names = lib.row_column_names(row_ptr)
    cdef np.uint32_t n_columns = lib.n_columns(row_ptr)

    # build columns
    cdef np.ndarray columns = np.zeros(shape=n_columns, dtype=object)

    cdef np.uint32_t i
    for i in range(0, n_columns):
        columns[i] = row_col_names[i].decode()

    # np.ndarray would force all internal arrays to be object dtypes
    # b/c each array has a different dtype.
    cdef list output = []

    # Begin looping until no rows are returned
    cdef np.uint32_t row_idx = 0
    cdef np.uint32_t one = 1
    cdef np.uint32_t n_increment = 1_000
    cdef np.uint32_t current_array_len = 0
    cdef lib.RowDataArrayPtr row_data_ptr
    cdef lib.Data data
    cdef np.uint32_t row_len = columns.shape[0]
    while True:
        if row_ptr == NULL:
            break
        else:

            # Get and insert new row
            lib.row_data(row_ptr, row_data_array_ptr)
            if row_data_array_ptr == NULL:
                raise TypeError(f"Unable to pull row data, likely an unsupported type. Check stderr output.")

            if row_idx == 0:
                # Initialize arrays for output
                # will resize at `n_increment` if `n_rows` is not set.
                for i in range(0, n_columns):
                    data = lib.index_row(row_data_array_ptr, row_len, i)
                    output.append(
                        array_init(data, n_increment if n_rows == -1 else n_rows)
                    )

            # grow arrays if next insert is passed current len
            if _n_rows == -1 and current_array_len <= row_idx:
                    for i in range(0, n_columns):
                        resize(output[i], current_array_len + n_increment)
                    current_array_len += n_increment

            for i in range(0, n_columns):
                data = lib.index_row(row_data_array_ptr, row_len, i)
                output[i] = insert_data_into_array(data, output[i], row_idx)

            lib.free_row(row_ptr)
            row_idx += one

        row_ptr = lib.next_row(row_iterator)

    # Ensure arrays are correct size; only if n_rows not set
    if _n_rows == -1 and current_array_len != row_idx:
        for i in range(0, n_columns):
            resize(output[i], row_idx)

    lib.free_row_data_array(row_data_array_ptr, row_len)
    lib.free_row_iter(row_iterator)
    lib.free_row_column_names(row_col_names)

    return {columns[i]: output[i] for i in range(columns.shape[0])}

cdef int resize(np.ndarray arr, int len) except -1:
    cdef int refcheck = 0
    cdef np.PyArray_Dims dims;
    cdef np.npy_intp dims_arr[1]
    dims_arr[0] = <np.npy_intp>len
    dims.ptr = <np.npy_intp*>&dims_arr
    dims.len = 1
    cdef object ret
    ret = np.PyArray_Resize(arr, &dims, refcheck, np.NPY_CORDER)
    if ret is not None:
        return -1
    else:
        return 0


cdef np.ndarray array_init(lib.Data data, int len):
    cdef np.ndarray array
    if data.tag == lib.Data_Tag.Int8:
        array = np.empty(shape=len, dtype=np.int8)
    elif data.tag == lib.Data_Tag.Int16:
        array = np.empty(shape=len, dtype=np.int16)
    elif data.tag == lib.Data_Tag.Uint32:
        array = np.empty(shape=len, dtype=np.uint32)
    elif data.tag == lib.Data_Tag.Int32:
        array = np.empty(shape=len, dtype=np.int32)
    elif data.tag == lib.Data_Tag.Int64:
        array = np.empty(shape=len, dtype=np.int64)
    elif data.tag == lib.Data_Tag.Float32:
        array = np.empty(shape=len, dtype=np.float32)
    elif data.tag == lib.Data_Tag.Float64:
        array = np.empty(shape=len, dtype=np.float64)
    elif data.tag == lib.Data_Tag.String:
        array = np.empty(shape=len, dtype=object)
    elif data.tag == lib.Data_Tag.Boolean:
        array = np.empty(shape=len, dtype=bool)
    elif data.tag == lib.Data_Tag.Bytes:
        array = np.empty(shape=len, dtype=bytearray)
    elif data.tag == lib.Data_Tag.Decimal:
        array = np.empty(shape=len, dtype=np.float64)
    elif data.tag == lib.Data_Tag.Null:
        array = np.empty(shape=len, dtype=object)
    else:
        raise ValueError(f"Unsupported tag: {data.tag}")
    return array


cdef np.ndarray insert_data_into_array(lib.Data data, np.ndarray arr, int idx):
    cdef np.ndarray[np.uint8_t, ndim=1] arr_bytes
    cdef np.npy_intp intp

    if data.tag == lib.Data_Tag.Boolean:
        arr[idx] = data.boolean._0

    elif data.tag == lib.Data_Tag.Bytes:
        intp = <np.npy_intp>data.bytes._0.len
        arr_bytes = np.PyArray_SimpleNewFromData(1, &intp, np.NPY_UINT8, data.bytes._0.ptr)
        PyArray_ENABLEFLAGS(arr_bytes, np.NPY_OWNDATA)
        arr[idx] = arr_bytes

    elif data.tag == lib.Data_Tag.Int8:
        arr[idx] = data.int8._0

    elif data.tag == lib.Data_Tag.Int16:
        arr[idx] = data.int16._0

    elif data.tag == lib.Data_Tag.Uint32:
        arr[idx] = data.uint32._0

    elif data.tag == lib.Data_Tag.Int64:
        arr[idx] = data.int64._0

    elif data.tag == lib.Data_Tag.Int32:
        arr[idx] = data.int32._0

    elif data.tag == lib.Data_Tag.Float64:
        arr[idx] = data.float64._0

    elif data.tag == lib.Data_Tag.Float32:
        arr[idx] = data.float32._0

    elif data.tag == lib.Data_Tag.String:
        arr[idx] = data.string._0.decode()

    elif data.tag == lib.Data_Tag.Decimal:
        arr[idx] = data.decimal._0

    elif data.tag == lib.Data_Tag.Null:
        if arr.dtype != object:
            arr = arr.astype(object, copy=False)
        arr[idx] = None

    else:
        raise ValueError(f"Unsupported Data enum {data.tag}")
    return arr

cdef class Database:

    cdef lib.DatabasePtr db_ptr
    cdef bytes uri

    def __init__(self, str uri):
        self.uri = uri.encode("utf-8")
        self._create_db()

    cdef _create_db(self):
        self.db_ptr = <lib.DatabasePtr>malloc(sizeof(np.uint32_t))
        self.db_ptr = lib.db_create(<char*>self.uri)

    cpdef connect(self):
        lib.db_connect(self.db_ptr)

    cpdef disconnect(self):
        lib.db_disconnect(self.db_ptr)

    cpdef __enter__(self):
        self.connect()
        return self

    cpdef __exit__(self, type, value, traceback):
        self.disconnect()

    def __dealloc__(self):
        if &self.db_ptr != NULL:
            lib.drop(self.db_ptr)
