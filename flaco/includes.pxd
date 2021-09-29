cimport numpy as np

cdef extern from "./libflaco.h":

    ctypedef enum Data_Tag:
        Int64
        Float64
        String

    ctypedef struct Int64_Body:
        np.int64_t _0

    ctypedef struct Float64_Body:
        np.float64_t _0

    ctypedef struct String_Body:
        const char *_0

    ctypedef struct Data:
        Data_Tag tag
        Int64_Body int64
        Float64_Body float64
        String_Body string

    Data read_sql(char *stmt_ptr, np.uint32_t *engine_ptr)
    np.uint32_t* create_engine(char *uri_ptr)
    void free_engine(np.uint32_t *engine_ptr)
