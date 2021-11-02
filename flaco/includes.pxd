# distutils: language=c
# cython: language_level=3

cimport numpy as np
from libcpp cimport bool


cdef extern from "./flaco.h":

    ctypedef struct BytesPtr:
        np.uint8_t *ptr
        np.uint32_t len

    ctypedef struct StringPtr:
        char *ptr
        np.uint32_t len

    ctypedef struct DateInfo:
        np.int8_t year
        np.uint8_t month
        np.uint8_t day

    ctypedef struct TimeInfo:
        np.uint8_t hour
        np.uint8_t minute
        np.uint8_t second
        np.uint32_t usecond

    ctypedef struct DateTimeInfo:
        DateInfo date
        TimeInfo time

    ctypedef struct TzInfo:
        np.int8_t hours
        np.int8_t minutes
        np.int8_t seconds
        bool is_positive

    ctypedef struct DateTimeTzInfo:
        DateInfo date
        TimeInfo time
        TzInfo tz

    ctypedef enum Data_Tag:
        Bytes
        Boolean
        Date
        DateTime
        DateTimeTz
        Time
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

    ctypedef char *Exception;

    ctypedef struct Bytes_Body:
        BytesPtr *_0

    ctypedef struct Boolean_Body:
        bool *_0

    ctypedef struct Date_Body:
        DateInfo *_0

    ctypedef struct DateTime_Body:
        DateTimeInfo *_0

    ctypedef struct DateTimeTz_Body:
        DateTimeTzInfo *_0

    ctypedef struct Time_Body:
        TimeInfo *_0

    ctypedef struct Decimal_Body:
        np.float64_t *_0

    ctypedef struct Int8_Body:
        np.int8_t *_0

    ctypedef struct Int16_Body:
        np.int16_t *_0

    ctypedef struct Uint32_Body:
        np.uint32_t *_0

    ctypedef struct Int32_Body:
        np.int32_t *_0

    ctypedef struct Int64_Body:
        np.int64_t *_0

    ctypedef struct Float32_Body:
        np.float32_t *_0

    ctypedef struct Float64_Body:
        np.float64_t *_0

    ctypedef struct String_Body:
        StringPtr *_0

    ctypedef struct Data:
        Data_Tag tag

        Bytes_Body bytes
        Boolean_Body boolean
        Date_Body date
        DateTime_Body date_time
        DateTimeTz_Body date_time_tz
        Time_Body time
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
    ctypedef char **RowColumnNamesArrayPtr
    ctypedef np.uint32_t *RowDataArrayPtr
    ctypedef np.uint32_t *SessionPtr

    DatabasePtr db_create(char *uri_ptr)
    void db_connect(DatabasePtr ptr, Exception *exc)
    void db_disconnect(DatabasePtr ptr)
    SessionPtr init_session()
    void free_session(SessionPtr session)
    RowIteratorPtr read_sql(const char *stmt_ptr, DatabasePtr db_ptr, Exception *exc)

    void free_db(DatabasePtr ptr)

    void next_row(
            RowIteratorPtr *row_iter_ptr,
            RowDataArrayPtr *row_ptr,
            np.uint32_t *n_columns,
            RowColumnNamesArrayPtr *column_names,
            SessionPtr *session,
            Exception *exc
    )

    Data *index_row(RowDataArrayPtr row_ptr, np.uint32_t len, np.uint32_t idx)
