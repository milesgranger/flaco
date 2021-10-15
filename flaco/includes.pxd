# distutils: language=c
# cython: language_level=3

cimport numpy as np
from libcpp cimport bool


cdef extern from "./flaco.h":

    ctypedef struct BytesPtr:
        np.uint8_t *ptr
        np.uint32_t len

    ctypedef enum Data_Tag:
        Bytes
        Boolean
        Decimal
        Int8
        Int16
        Uint32
        Int32
        Int64
        Float32
        Float64
        String
        Null

    ctypedef char **Exception;

    ctypedef struct Bytes_Body:
        BytesPtr _0

    ctypedef struct Boolean_Body:
        bool _0

    ctypedef struct Decimal_Body:
        np.float64_t _0

    ctypedef struct Int8_Body:
        np.int8_t _0

    ctypedef struct Int16_Body:
        np.int16_t _0

    ctypedef struct Uint32_Body:
        np.uint32_t _0

    ctypedef struct Int32_Body:
        np.int32_t _0

    ctypedef struct Int64_Body:
        np.int64_t _0

    ctypedef struct Float32_Body:
        np.float32_t _0

    ctypedef struct Float64_Body:
        np.float64_t _0

    ctypedef struct String_Body:
        const char *_0

    ctypedef struct Data:
        Data_Tag tag

        Bytes_Body bytes
        Boolean_Body boolean
        Decimal_Body decimal
        Int8_Body int8
        Int16_Body int16
        Uint32_Body uint32
        Int32_Body int32
        Int64_Body int64
        Float32_Body float32
        Float64_Body float64
        String_Body string

    ctypedef np.uint32_t *DatabasePtr
    ctypedef np.uint32_t *RowIteratorPtr
    ctypedef np.uint32_t *RowPtr
    ctypedef char **RowColumnNamesArrayPtr
    ctypedef np.uint32_t *RowDataArrayPtr

    DatabasePtr db_create(char *uri_ptr)
    void db_connect(DatabasePtr ptr)
    void db_disconnect(DatabasePtr ptr)

    RowIteratorPtr read_sql(const char *stmt_ptr, DatabasePtr db_ptr, Exception exc)
    void free_row_iter(RowIteratorPtr ptr);

    void drop(DatabasePtr ptr)

    np.uint32_t n_columns(RowPtr row_ptr)

    RowPtr next_row(RowIteratorPtr row_iter_ptr)
    void free_row(RowPtr ptr);

    RowColumnNamesArrayPtr row_column_names(RowPtr row_ptr)
    void free_row_column_names(RowColumnNamesArrayPtr ptr)

    RowDataArrayPtr init_row_data_array(RowPtr row_ptr)

    void row_data(RowPtr row_ptr, RowDataArrayPtr array_ptr, Exception exc)
    void free_row_data_array(RowDataArrayPtr ptr, np.uint32_t len)

    Data index_row(RowPtr row_ptr, np.uint32_t len, np.uint32_t idx)
