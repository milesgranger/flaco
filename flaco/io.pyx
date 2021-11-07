# distutils: language=c
# cython: language_level=3, boundscheck=False

cimport numpy as np
import numpy as np
import datetime as dt
from libc.stdlib cimport malloc, free
cimport cpython.datetime as dt
from cython.operator cimport dereference as deref
from flaco cimport includes as lib


np.import_array()
dt.import_datetime()

cdef extern from "Python.h":
    object PyUnicode_InternFromString(char *v)


cdef:
    np.int64_t DATE_01_JAN_2000 = np.datetime64('2000-01-01', 'D').astype(np.int64)
    np.int64_t DATETIME_MID_NIGHT_01_JAN_2000 = np.datetime64('2000-01-01T00:00:00', 'us').astype(np.int64)

cpdef dict read_sql(str stmt, Database db, int n_rows=-1, int size_hint=-1):
    cdef:
        bytes stmt_bytes = stmt.encode("utf-8")
        np.int32_t _n_rows = n_rows
        lib.RowDataArrayPtr row_data_array_ptr = NULL
        lib.RowColumnNamesArrayPtr column_names = NULL
        np.uint32_t n_columns = 0
        lib.Exception exc = NULL

    cdef lib.RowIteratorPtr row_iterator = lib.read_sql(
        <char*>stmt_bytes, db.db_ptr, &exc
    )
    if exc != NULL:
        raise FlacoException(exc.decode())

    # Read first row
    lib.next_row(
        &row_iterator,
        &row_data_array_ptr,
        &n_columns,
        &column_names,
        &exc
    )
    if exc != NULL:
        raise FlacoException(exc.decode())

    # build columns
    cdef np.ndarray columns = np.zeros(shape=n_columns, dtype=object)
    cdef np.uint32_t i
    for i in range(0, n_columns):
        columns[i] = column_names[i].decode()

    # np.ndarray would force all internal arrays to be object dtypes
    # b/c each array has a different dtype.
    cdef list output = []

    # Begin looping until no rows are returned
    cdef:
        np.uint32_t row_idx     = 0
        np.uint32_t one         = 1
        np.uint32_t n_increment = 1_000 if size_hint == -1 else size_hint
        np.uint32_t current_array_len = 0
        lib.RowDataArrayPtr row_data_ptr
        lib.Data *data

    while True:
        if row_iterator == NULL:
            break
        else:

            if row_idx == 0:
                # Initialize arrays for output
                # will resize at `n_increment` if `n_rows` is not set.
                for i in range(0, n_columns):
                    data = lib.index_row(row_data_array_ptr, n_columns, i)
                    output.append(
                        array_init(deref(data), n_increment if n_rows == -1 else n_rows)
                    )

            # grow arrays if next insert is passed current len
            if _n_rows == -1 and current_array_len <= row_idx:
                    for i in range(0, n_columns):
                        resize(output[i], current_array_len + n_increment)
                    current_array_len += n_increment

            for i in range(0, n_columns):
                data = lib.index_row(row_data_array_ptr, n_columns, i)
                output[i] = insert_data_into_array(deref(data), output[i], row_idx)

            row_idx += one

            lib.next_row(
                &row_iterator,
                &row_data_array_ptr,
                &n_columns,
                &column_names,
                &exc
            )
            if exc != NULL:
                raise FlacoException(exc.decode())

    # Ensure arrays are correct size; only if n_rows not set
    if _n_rows == -1 and current_array_len != row_idx:
        for i in range(0, n_columns):
            resize(output[i], row_idx)
    return {columns[i]: output[i] for i in range(n_columns)}

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
    elif data.tag == lib.Data_Tag.Date:
        array = np.empty(shape=len, dtype="datetime64[D]")
    elif data.tag == lib.Data_Tag.DateTime:
        array = np.empty(shape=len, dtype="datetime64[us]")
    elif data.tag == lib.Data_Tag.Time:
        array = np.empty(shape=len, dtype=dt.time)
    else:
        raise ValueError(f"Unsupported tag: {data.tag}")
    return array


cdef np.ndarray insert_data_into_array(lib.Data data, np.ndarray arr, int idx):
    cdef:
        np.ndarray[np.uint8_t, ndim=1] arr_bytes
        np.npy_intp intp

    if data.tag == lib.Data_Tag.Boolean:
        arr[idx] = data.boolean._0

    elif data.tag == lib.Data_Tag.Bytes:
        arr[idx] = data.bytes._0.ptr[:data.bytes._0.len]
        free(data.bytes._0.ptr)

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
        arr[idx] = PyUnicode_InternFromString(<char*>data.string._0.ptr)
        free(data.string._0.ptr)

    elif data.tag == lib.Data_Tag.Date:
        arr[idx] = np.timedelta64(
            DATE_01_JAN_2000 + data.date._0.offset,
            'D'
        )

    elif data.tag == lib.Data_Tag.DateTime:
        arr[idx] = np.timedelta64(
            DATETIME_MID_NIGHT_01_JAN_2000 + data.date_time._0.offset,
            'us'
        )

    elif data.tag == lib.Data_Tag.Time:
        arr[idx] = dt.time_new(
            data.time._0.hour,
            data.time._0.minute,
            data.time._0.second,
            data.time._0.usecond,
            None
        )

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
        cdef lib.Exception exc = NULL
        lib.db_connect(self.db_ptr, &exc)
        if exc != NULL:
            raise FlacoException(exc.decode())

    cpdef disconnect(self):
        lib.db_disconnect(self.db_ptr)

    cpdef __enter__(self):
        self.connect()
        return self

    cpdef __exit__(self, type, value, traceback):
        self.disconnect()

    def __dealloc__(self):
        if &self.db_ptr != NULL:
            lib.free_db(self.db_ptr)


cdef class FlacoException(Exception):
    pass
