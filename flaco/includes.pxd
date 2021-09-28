cimport numpy as np

cdef extern from "./libflaco.h":

    int read_sql()
