cimport numpy as np

cdef extern from "./libflaco.h":

    ctypedef enum Data_Tag:
        Int64

    ctypedef struct Int64_Body:
        np.int64_t _0

    ctypedef struct Data:
        Data_Tag tag
        Int64_Body int64

    Data read_sql()
