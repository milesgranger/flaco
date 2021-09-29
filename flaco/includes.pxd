cimport numpy as np
from libcpp cimport bool

cdef extern from "./libflaco.h":

    ctypedef enum Data_Tag:
        Int64
        Int32
        Float64
        String

    ctypedef struct Int32_Body:
        np.int32_t _0

    ctypedef struct Int64_Body:
        np.int64_t _0

    ctypedef struct Float64_Body:
        np.float64_t _0

    ctypedef struct String_Body:
        const char *_0

    ctypedef struct Data:
        Data_Tag tag
        Int32_Body int32
        Int64_Body int64
        Float64_Body float64
        String_Body string

    ctypedef np.uint32_t *RowIteratorPtr
    ctypedef np.uint32_t *RowPtr
    ctypedef char **RowColumnNamesArrayPtr
    ctypedef np.uint32_t *RowDataArrayPtr
    ctypedef char **RowTypesArrayPtr

    RowIteratorPtr read_sql(char *stmt_ptr, np.uint32_t *engine_ptr)

    np.uint32_t* create_engine(char *uri_ptr)

    void drop(np.uint32_t *ptr)

    np.uint32_t n_columns(RowPtr row_ptr)

    RowPtr next_row(RowIteratorPtr row_iter_ptr)

    RowIteratorPtr read_sql(const char *stmt_ptr, np.uint32_t *engine_ptr)

    RowColumnNamesArrayPtr row_column_names(RowPtr row_ptr)

    RowDataArrayPtr row_data(RowPtr row_ptr, RowTypesArrayPtr row_types_ptr)

    RowTypesArrayPtr row_types(RowPtr row_ptr)
