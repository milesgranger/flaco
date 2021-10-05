cimport numpy as np


cdef extern from "./libflaco.h":

    ctypedef enum Data_Tag:
        Int32
        Int64
        Float32
        Float64
        String
        Null

    ctypedef struct Int32_Body:
        const np.int32_t _0

    ctypedef struct Int64_Body:
        const np.int64_t _0

    ctypedef struct Float32_Body:
        np.float32_t _0

    ctypedef struct Float64_Body:
        np.float64_t _0

    ctypedef struct String_Body:
        const char *_0

    ctypedef struct Data:
        Data_Tag tag

        Int32_Body int32
        Int64_Body int64
        Float32_Body float32
        Float64_Body float64
        String_Body string

    ctypedef np.uint32_t *RowIteratorPtr
    ctypedef np.uint32_t *RowPtr
    ctypedef char **RowColumnNamesArrayPtr
    ctypedef np.uint32_t *RowDataArrayPtr

    RowIteratorPtr read_sql(char *stmt_ptr, np.uint32_t *engine_ptr)

    np.uint32_t* create_connection(char *uri_ptr)

    void drop(np.uint32_t *ptr)

    np.uint32_t n_columns(RowPtr row_ptr)

    RowPtr next_row(RowIteratorPtr row_iter_ptr)
    void free_row(RowPtr ptr);

    RowIteratorPtr read_sql(const char *stmt_ptr, np.uint32_t *engine_ptr)
    void free_row_iter(RowIteratorPtr ptr);

    RowColumnNamesArrayPtr row_column_names(RowPtr row_ptr)
    void free_row_column_names(RowColumnNamesArrayPtr ptr)

    RowDataArrayPtr row_data(RowPtr row_ptr)
    void free_row_data_array(RowDataArrayPtr ptr)

    Data index_row(RowPtr row_ptr, np.uint32_t idx)
