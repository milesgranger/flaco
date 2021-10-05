//#![warn(missing_docs)]
use postgres as pg;
use postgres::fallible_iterator::FallibleIterator;
use postgres::RowIter;
use std::os::raw::c_char;
use std::{ffi, mem};

type RowIteratorPtr = *mut u32;
type RowPtr = *mut u32;
type RowDataArrayPtr = *mut u32;
type RowColumnNamesArrayPtr = *const *const c_char;

/// Supports creating connections to a given connection URI
pub struct Connection {
    client: pg::Client,
}

impl Connection {
    /// Create an `Connection` object
    /// `uri` is to conform to any of the normal connection strings, described
    /// in more [detail here](https://docs.rs/tokio-postgres/0.7.2/tokio_postgres/config/struct.Config.html#examples)
    pub fn new(uri: &str) -> Self {
        let client = pg::Client::connect(uri, pg::NoTls).unwrap();
        Self { client }
    }
}

#[derive(Clone, Debug)]
#[repr(C)]
pub struct BytesPtr {
    ptr: *const u8,
    len: u32
}

#[derive(Clone, Debug)]
#[repr(C)]
pub enum Data {
    Bytes(BytesPtr),
    Boolean(bool),
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Uint32(u32),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    String(*const c_char),
    Null,
}

#[no_mangle]
pub extern "C" fn create_connection(uri_ptr: *const c_char) -> *mut u32 {
    let uri_c = unsafe { ffi::CStr::from_ptr(uri_ptr) };
    let uri = uri_c.to_str().unwrap();
    let con = Box::new(Connection::new(uri));
    Box::into_raw(con) as *mut _
}

#[no_mangle]
pub extern "C" fn drop(ptr: *mut u32) {
    unsafe { Box::from_raw(ptr) };
}

#[no_mangle]
pub extern "C" fn read_sql(stmt_ptr: *const c_char, con_ptr: *mut u32) -> RowIteratorPtr {
    let mut con = unsafe { Box::from_raw(con_ptr as *mut Connection) };
    let stmt_c = unsafe { ffi::CStr::from_ptr(stmt_ptr) };
    let stmt = stmt_c.to_str().unwrap();
    let row_iter = con.client.query_raw::<_, &i32, _>(stmt, &[]).unwrap();
    // read query to start rowstream

    // get first row, and construct schema/columns in numpy

    // iterate over each row in the stream

    // for each column value in row

    // First iteration, check if arrays should be resized to fit new row.

    // if value is None, convert to appropriate pandas null type (pd.NA, pd.NaT)

    // insert element into array
    let row_iterator = Box::new(row_iter);
    let ptr = Box::into_raw(row_iterator) as RowIteratorPtr;
    mem::forget(con);
    ptr
}
#[no_mangle]
pub extern "C" fn free_row_iter(ptr: RowIteratorPtr) {
    let _ = unsafe { Box::from_raw(ptr as *mut pg::RowIter) };
}

#[no_mangle]
pub extern "C" fn next_row(row_iter_ptr: RowIteratorPtr) -> RowPtr {
    let mut row_iter = unsafe { Box::from_raw(row_iter_ptr as *mut RowIter) };
    let ptr = match row_iter.next().unwrap() {
        Some(row) => Box::into_raw(Box::new(row)) as RowPtr,
        None => std::ptr::null_mut(),
    };
    mem::forget(row_iter);
    ptr
}
#[no_mangle]
pub extern "C" fn free_row(ptr: RowPtr) {
    let _ = unsafe { Box::from_raw(ptr as *mut pg::Row) };
}

#[no_mangle]
pub extern "C" fn row_data(row_ptr: RowPtr) -> RowDataArrayPtr {
    let row = unsafe { Box::from_raw(row_ptr as *mut pg::Row) };
    let len = row.len();
    let mut values = Vec::with_capacity(len);
    for i in 0..len {
        let type_ = row.columns()[i].type_();
        // TODO: postgres-types: expose Inner enum which these variations
        // and have postgres Row.type/(or something) expose the variant
        println!("{}", type_.name());
        let val = match type_.name() {
            "bytea" => {
                let val: Option<Vec<u8>> = row.get(i);
                match val {
                    Some(v) => {
                        let ptr = v.as_ptr();
                        let len = v.len() as u32;
                        mem::forget(v);
                        Data::Bytes(BytesPtr { ptr, len })
                    },
                    None => Data::Null
                }
            }
            "char" => {
                let val: Option<i8> = row.get(i);
                match val {
                    Some(v) => Data::Int8(v),
                    None => Data::Null
                }
            }
            "smallint" | "smallserial" | "int2" => {
                let val: Option<i16> = row.get(i);
                match val {
                    Some(v) => Data::Int16(v),
                    None => Data::Null
                }
            }
            "oid" => {
                let val: Option<u32> = row.get(i);
                match val {
                    Some(v) => Data::Uint32(v),
                    None => Data::Null
                }
            }
            "int4" | "int" | "serial" => {
                let val: Option<i32> = row.get(i);
                match val {
                    Some(v) => Data::Int32(v),
                    None => Data::Null,
                }
            }
            "bigint" | "int8" | "bigserial" => {
                let val: Option<i64> = row.get(i);
                match val {
                    Some(v) => Data::Int64(v),
                    None => Data::Null,
                }
            }
            "bool" => {
                let val: Option<bool> = row.get(i);
                match val {
                    Some(v) => Data::Boolean(v),
                    None => Data::Null
                }
            }
            "double precision" | "float8" => {
                let val: Option<f64> = row.get(i);
                Data::Float64(val.unwrap_or_else(|| f64::NAN))
            }
            "real" => {
                let val: Option<f32> = row.get(i);
                Data::Float32(val.unwrap_or_else(|| f32::NAN))
            }
            _ => {
                let string: Option<String> = row.get(i);
                let ptr = match string {
                    Some(string) => {
                        let cstring = ffi::CString::new(string).unwrap();
                        let ptr = cstring.as_ptr();
                        mem::forget(cstring);
                        ptr
                    }
                    None => std::ptr::null(),
                };
                Data::String(ptr)
            }
        };
        values.push(val)
    }
    mem::forget(row);
    Box::into_raw(Box::new(values)) as RowDataArrayPtr
}

#[no_mangle]
pub extern "C" fn free_row_data_array(ptr: RowDataArrayPtr) {
    let _ = unsafe { Box::from_raw(ptr as *mut Vec<Data>) };
}

#[no_mangle]
pub extern "C" fn n_columns(row_ptr: RowPtr) -> u32 {
    let row = unsafe { Box::from_raw(row_ptr as *mut pg::Row) };
    let len = row.len() as u32;
    mem::forget(row);
    len
}

#[no_mangle]
pub extern "C" fn row_column_names(row_ptr: RowPtr) -> RowColumnNamesArrayPtr {
    let row = unsafe { Box::from_raw(row_ptr as *mut pg::Row) };
    let names = row
        .columns()
        .iter()
        .map(|col| col.name())
        .map(|name| ffi::CString::new(name).unwrap())
        .map(|name| {
            let ptr = name.as_ptr();
            mem::forget(name);
            ptr
        })
        .collect::<Vec<*const c_char>>();
    mem::forget(row);
    let ptr = names.as_ptr();
    mem::forget(names);
    ptr
}

#[no_mangle]
pub extern "C" fn free_row_column_names(ptr: RowColumnNamesArrayPtr) {
    let _names = unsafe { Box::from_raw(ptr as *mut Vec<*const c_char>) };
}

#[no_mangle]
pub extern "C" fn index_row(row_ptr: RowDataArrayPtr, idx: u32) -> Data {
    let row = unsafe { Box::from_raw(row_ptr as *mut Vec<Data>) };
    let data = row[idx as usize].clone();
    mem::forget(row);
    data
}

#[cfg(test)]
mod tests {

    use super::*;
    const CONNECTION_URI: &str = "postgresql://postgres:postgres@localhost:5432/postgres";

    fn basic_query() {
        let con = Connection::new(CONNECTION_URI);
        con.execute("create table if not exists foobar (col1 integer, col2 integer)");
        let n_rows = con.execute("insert into foobar (col1, col2) values (1, 1)");
        assert_eq!(n_rows, 1)
    }
}
