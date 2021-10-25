/* Binding from Rust to Cython  */

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef uint32_t *DatabasePtr;

typedef char *Exception;

typedef struct {
  uint8_t *ptr;
  uint32_t len;
} BytesPtr;

typedef struct {
  int32_t year;
  uint8_t month;
  uint8_t day;
} DateInfo;

typedef struct {
  uint8_t hour;
  uint8_t minute;
  uint8_t second;
  uint32_t usecond;
} TimeInfo;

typedef struct {
  DateInfo date;
  TimeInfo time;
} DateTimeInfo;

typedef struct {
  int8_t hours;
  int8_t minutes;
  bool is_positive;
} TzInfo;

typedef struct {
  DateInfo date;
  TimeInfo time;
  TzInfo tz;
} DateTimeTzInfo;

typedef struct {
  char *ptr;
  uint32_t len;
} StringPtr;

typedef enum {
  Bytes,
  Date,
  DateTime,
  DateTimeTz,
  Time,
  Boolean,
  Decimal,
  Int8,
  Int16,
  Int32,
  Uint32,
  Int64,
  Float32,
  Float64,
  String,
  Null,
} Data_Tag;

typedef struct {
  BytesPtr _0;
} Bytes_Body;

typedef struct {
  DateInfo _0;
} Date_Body;

typedef struct {
  DateTimeInfo _0;
} DateTime_Body;

typedef struct {
  DateTimeTzInfo _0;
} DateTimeTz_Body;

typedef struct {
  TimeInfo _0;
} Time_Body;

typedef struct {
  bool _0;
} Boolean_Body;

typedef struct {
  double _0;
} Decimal_Body;

typedef struct {
  int8_t _0;
} Int8_Body;

typedef struct {
  int16_t _0;
} Int16_Body;

typedef struct {
  int32_t _0;
} Int32_Body;

typedef struct {
  uint32_t _0;
} Uint32_Body;

typedef struct {
  int64_t _0;
} Int64_Body;

typedef struct {
  float _0;
} Float32_Body;

typedef struct {
  double _0;
} Float64_Body;

typedef struct {
  StringPtr _0;
} String_Body;

typedef struct {
  Data_Tag tag;
  union {
    Bytes_Body bytes;
    Date_Body date;
    DateTime_Body date_time;
    DateTimeTz_Body date_time_tz;
    Time_Body time;
    Boolean_Body boolean;
    Decimal_Body decimal;
    Int8_Body int8;
    Int16_Body int16;
    Int32_Body int32;
    Uint32_Body uint32;
    Int64_Body int64;
    Float32_Body float32;
    Float64_Body float64;
    String_Body string;
  };
} Data;

typedef uint32_t *RowDataArrayPtr;

typedef uint32_t *RowIteratorPtr;

typedef char **RowColumnNamesArrayPtr;

void db_connect(DatabasePtr ptr, Exception *exc);

DatabasePtr db_create(const char *uri_ptr);

void db_disconnect(DatabasePtr ptr);

void free_db(DatabasePtr ptr);

Data *index_row(RowDataArrayPtr row_data_array_ptr, uint32_t len, uint32_t idx);

void next_row(RowIteratorPtr *row_iter_ptr,
              RowDataArrayPtr *row_data_array_ptr,
              uint32_t *n_columns,
              RowColumnNamesArrayPtr *column_names,
              Exception *exc);

RowIteratorPtr read_sql(const char *stmt_ptr, DatabasePtr db_ptr, Exception *exc);
