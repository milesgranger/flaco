/* Binding from Rust to Cython  */

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef uint32_t *DatabasePtr;

typedef uint32_t *RowPtr;

typedef char *const *RowColumnNamesArrayPtr;

typedef uint32_t *RowDataArrayPtr;

typedef uint32_t *RowIteratorPtr;

typedef struct {
  uint8_t *ptr;
  uint32_t len;
} BytesPtr;

typedef enum {
  Bytes,
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
  const char *_0;
} String_Body;

typedef struct {
  Data_Tag tag;
  union {
    Bytes_Body bytes;
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

void db_connect(DatabasePtr ptr);

DatabasePtr db_create(const char *uri_ptr);

void db_disconnect(DatabasePtr ptr);

void drop(uint32_t *ptr);

void free_row(RowPtr ptr);

void free_row_column_names(RowColumnNamesArrayPtr ptr);

void free_row_data_array(RowDataArrayPtr ptr, uint32_t len);

void free_row_iter(RowIteratorPtr ptr);

Data index_row(RowDataArrayPtr row_ptr, uint32_t len, uint32_t idx);

RowDataArrayPtr init_row_data_array(RowPtr row_ptr);

uint32_t n_columns(RowPtr row_ptr);

RowPtr next_row(RowIteratorPtr row_iter_ptr);

RowIteratorPtr read_sql(const char *stmt_ptr, DatabasePtr db_ptr);

RowColumnNamesArrayPtr row_column_names(RowPtr row_ptr);

void row_data(RowPtr row_ptr, RowDataArrayPtr array_ptr);