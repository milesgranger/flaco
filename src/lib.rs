//#![warn(missing_docs)]
use postgres as pg;
use postgres::fallible_iterator::FallibleIterator;
use postgres::RowIter;
use std::os::raw::c_char;
use std::{ffi, mem};

type RowIteratorPtr = *mut u32;
type RowPtr = *const u32;
type RowDataArrayPtr = *const u32;
type RowTypesArrayPtr = *const u32;
type RowColumnNamesArrayPtr = *const u32;

/// Supports creating connections to a given connection URI
pub struct Engine {
    client: pg::Client,
}

impl Engine {
    /// Create an `Engine` object
    /// `uri` is to conform to any of the normal connection strings, described
    /// in more [detail here](https://docs.rs/tokio-postgres/0.7.2/tokio_postgres/config/struct.Config.html#examples)
    pub fn new(uri: &str) -> Self {
        let client = pg::Client::connect(uri, pg::NoTls).unwrap();
        Self { client }
    }
}

#[repr(C)]
pub enum Data {
    Int64(i64),
    Int32(i32),
    Float64(f64),
    String(*const c_char),
}

#[no_mangle]
pub extern "C" fn create_engine(uri_ptr: *const c_char) -> *mut u32 {
    let uri_c = unsafe { ffi::CStr::from_ptr(uri_ptr) };
    let uri = uri_c.to_str().unwrap();
    let engine = Box::new(Engine::new(uri));
    Box::into_raw(engine) as *mut _
}

#[no_mangle]
pub extern "C" fn drop(ptr: *mut u32) {
    unsafe { Box::from_raw(ptr) };
}

#[no_mangle]
pub extern "C" fn read_sql(stmt_ptr: *const c_char, engine_ptr: *mut u32) -> RowIteratorPtr {
    let mut engine = unsafe { Box::from_raw(engine_ptr as *mut Engine) };
    let stmt_c = unsafe { ffi::CStr::from_ptr(stmt_ptr) };
    let stmt = stmt_c.to_str().unwrap();
    let row_iter = engine.client.query_raw::<_, &i32, _>(stmt, &[]).unwrap();
    // read query to start rowstream

    // get first row, and construct schema/columns in numpy

    // iterate over each row in the stream

    // for each column value in row

    // First iteration, check if arrays should be resized to fit new row.

    // if value is None, convert to appropriate pandas null type (pd.NA, pd.NaT)

    // insert element into array
    let row_iterator = Box::new(row_iter);
    let ptr = Box::into_raw(row_iterator) as RowIteratorPtr;
    mem::forget(engine);
    ptr
}

#[no_mangle]
pub extern "C" fn next_row(row_iter_ptr: RowIteratorPtr) -> RowPtr {
    let mut row_iter = unsafe { Box::from_raw(row_iter_ptr as *mut RowIter) };
    let ptr = match row_iter.next().unwrap() {
        Some(row) => Box::into_raw(Box::new(row)) as RowPtr,
        None => std::ptr::null(),
    };
    mem::forget(row_iter);
    ptr
}

#[no_mangle]
pub extern "C" fn row_data(row_ptr: RowPtr, row_types_ptr: RowTypesArrayPtr) -> RowDataArrayPtr {
    let row = unsafe { Box::from_raw(row_ptr as *mut pg::Row) };
    let row_types = unsafe { Box::from_raw(row_types_ptr as *mut Vec<ffi::CString>) };
    let len = row.len();
    let mut values = Vec::with_capacity(len);
    for i in 0..len {
        let type_ = unsafe { row_types.get_unchecked(i) };
        let val = match type_.as_bytes() {
            b"int4" | b"int" | b"serial" => Data::Int32(row.get(i)),
            _ => unimplemented!("Unsupported type: {:?}", type_),
        };
        values.push(val)
    }
    mem::forget(row);
    mem::forget(row_types);
    Box::into_raw(Box::new(values)) as RowDataArrayPtr
}

#[no_mangle]
pub extern "C" fn row_types(row_ptr: RowPtr) -> RowTypesArrayPtr {
    let row = unsafe { Box::from_raw(row_ptr as *mut pg::Row) };
    let names = row
        .columns()
        .iter()
        .map(|col| col.type_().name().to_string())
        .map(|name| ffi::CString::new(name).unwrap())
        .collect::<Vec<ffi::CString>>();
    mem::forget(row);
    Box::into_raw(Box::new(names)) as RowTypesArrayPtr
}

#[no_mangle]
pub extern "C" fn row_column_names(row_ptr: RowPtr) -> RowColumnNamesArrayPtr {
    let row = unsafe { Box::from_raw(row_ptr as *mut pg::Row) };
    let names = row
        .columns()
        .iter()
        .map(|col| col.name())
        .map(|name| ffi::CString::new(name).unwrap())
        .collect::<Vec<ffi::CString>>();
    mem::forget(row);
    Box::into_raw(Box::new(names)) as RowColumnNamesArrayPtr
}

#[cfg(test)]
mod tests {

    use super::*;
    const CONNECTION_URI: &str = "postgresql://postgres:postgres@localhost:5432/postgres";

    fn basic_query() {
        let engine = Engine::new(CONNECTION_URI);
        engine.execute("create table if not exists foobar (col1 integer, col2 integer)");
        let n_rows = engine.execute("insert into foobar (col1, col2) values (1, 1)");
        assert_eq!(n_rows, 1)
    }
}
