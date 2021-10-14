//#![warn(missing_docs)]
use postgres as pg;
use postgres::fallible_iterator::FallibleIterator;
use postgres::RowIter;
use rust_decimal::prelude::{Decimal, ToPrimitive};
use std::net::IpAddr;
use std::os::raw::{c_char};
use std::{ffi, mem};
use time;
use time::format_description::well_known::Rfc3339;

type DatabasePtr = *mut u32;
type RowIteratorPtr = *mut u32;
type RowPtr = *mut u32;
type RowDataArrayPtr = *mut u32;
type RowColumnNamesArrayPtr = *const *mut c_char;

/// Supports creating connections to a given connection URI
pub struct Database {
    client: Option<pg::Client>,
    uri: String,
}

impl Database {
    /// Create an `Database` object
    /// `uri` is to conform to any of the normal connection strings, described
    /// in more [detail here](https://docs.rs/tokio-postgres/0.7.2/tokio_postgres/config/struct.Config.html#examples)
    pub fn new(uri: &str) -> Self {
        let client = None;
        Self {
            client,
            uri: uri.to_string(),
        }
    }

    pub fn connect(&mut self) {
        if self.client.is_none() {
            self.client = Some(pg::Client::connect(&self.uri, pg::NoTls).unwrap());
        }
    }

    pub fn disconnect(&mut self) {
        if self.client.is_some() {
            self.client = None;
        }
    }
}

#[no_mangle]
pub extern "C" fn db_create(uri_ptr: *const c_char) -> DatabasePtr {
    let uri_c = unsafe { ffi::CStr::from_ptr(uri_ptr) };
    let uri = uri_c.to_str().unwrap();
    let db = Box::new(Database::new(uri));
    Box::into_raw(db) as *mut _
}

#[no_mangle]
pub extern "C" fn read_sql(stmt_ptr: *const c_char, db_ptr: DatabasePtr) -> RowIteratorPtr {
    let mut db = unsafe { Box::from_raw(db_ptr as *mut Database) };
    let stmt_c = unsafe { ffi::CStr::from_ptr(stmt_ptr) };
    let stmt = stmt_c.to_str().unwrap();
    let row_iter = db
        .client
        .as_mut()
        .expect("Not connected!")
        .query_raw::<_, &i32, _>(stmt, &[])
        .unwrap();
    let boxed_row_iter = Box::new(row_iter);
    let ptr = Box::into_raw(boxed_row_iter) as RowIteratorPtr;
    mem::forget(db);
    ptr
}

#[no_mangle]
pub extern "C" fn db_disconnect(ptr: DatabasePtr) {
    let mut conn = unsafe { Box::from_raw(ptr as *mut Database) };
    conn.disconnect();
    mem::forget(conn);
}

#[no_mangle]
pub extern "C" fn db_connect(ptr: DatabasePtr) {
    let mut conn = unsafe { Box::from_raw(ptr as *mut Database) };
    conn.connect();
    mem::forget(conn);
}

#[derive(Clone, Debug)]
#[repr(C)]
pub struct BytesPtr {
    ptr: *mut u8,
    len: u32,
}

#[derive(Clone, Debug)]
#[repr(C)]
pub enum Data {
    Bytes(BytesPtr),
    Boolean(bool),
    Decimal(f64),  // TODO: support lossless decimal/numeric type handling
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

macro_rules! simple_from {
    ($t:ty, $i:ident) => {
        impl From<Option<$t>> for Data {
            fn from(val: Option<$t>) -> Self {
                match val {
                    Some(v) => Data::$i(v),
                    None => Data::Null,
                }
            }
        }
    };
}

simple_from!(bool, Boolean);
simple_from!(i8, Int8);
simple_from!(i16, Int16);
simple_from!(i32, Int32);
simple_from!(u32, Uint32);
simple_from!(i64, Int64);

impl From<Option<Vec<u8>>> for Data {
    fn from(val: Option<Vec<u8>>) -> Self {
        match val {
            Some(v) => {
                let ptr = v.as_ptr() as _;
                let len = v.len() as u32;
                mem::forget(v);
                Data::Bytes(BytesPtr { ptr, len })
            }
            None => Data::Null,
        }
    }
}
impl From<Option<String>> for Data {
    fn from(val: Option<String>) -> Self {
        match val {
            Some(string) => {
                let cstring = ffi::CString::new(string).unwrap();
                let ptr = cstring.as_ptr();
                mem::forget(cstring);
                Data::String(ptr)
            }
            None => Data::Null,
        }
    }
}

#[no_mangle]
pub extern "C" fn drop(ptr: *mut u32) {
    unsafe { Box::from_raw(ptr) };
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
        let val = match type_.name() {
            "bytea" => row.get::<_, Option<Vec<u8>>>(i).into(),
            "char" => row.get::<_, Option<i8>>(i).into(),
            "smallint" | "smallserial" | "int2" => row.get::<_, Option<i16>>(i).into(),
            "oid" => row.get::<_, Option<u32>>(i).into(),
            "int4" | "int" | "serial" => row.get::<_, Option<i32>>(i).into(),
            "bigint" | "int8" | "bigserial" => row.get::<_, Option<i64>>(i).into(),
            "bool" => row.get::<_, Option<bool>>(i).into(),
            "double precision" | "float8" => {
                let val: Option<f64> = row.get(i);
                Data::Float64(val.unwrap_or_else(|| f64::NAN))
            }
            "real" => {
                let val: Option<f32> = row.get(i);
                Data::Float32(val.unwrap_or_else(|| f32::NAN))
            }
            "varchar" | "char(n)" | "text" | "citext" | "name" | "unknown" | "bpchar" => {
                let string: Option<String> = row.get(i);
                Data::from(string)
            }
            "timestamp" => {
                let time_: Option<time::PrimitiveDateTime> = row.get(i);
                match time_ {
                    Some(t) => {
                        Data::from(Some(t.format(&Rfc3339).unwrap_or_else(|_| t.to_string())))
                    }
                    None => Data::Null,
                }
            }
            "timestamp with time zone" | "timestamptz" => {
                let time_: Option<time::OffsetDateTime> = row.get(i);
                match time_ {
                    Some(t) => {
                        Data::from(Some(t.format(&Rfc3339).unwrap_or_else(|_| t.to_string())))
                    }
                    None => Data::Null,
                }
            }
            "date" => {
                let time_: Option<time::Date> = row.get(i);
                match time_ {
                    Some(t) => {
                        Data::from(Some(t.format(&Rfc3339).unwrap_or_else(|_| t.to_string())))
                    }
                    None => Data::Null,
                }
            }
            "time" => {
                let time_: Option<time::Time> = row.get(i);
                match time_ {
                    Some(t) => {
                        Data::from(Some(t.format(&Rfc3339).unwrap_or_else(|_| t.to_string())))
                    }
                    None => Data::Null,
                }
            }
            "json" | "jsonb" => {
                let json: Option<serde_json::Value> = row.get(i);
                match json {
                    Some(j) => Data::from(Some(j.to_string())),
                    None => Data::Null,
                }
            }
            "uuid" => {
                let uuid_: Option<uuid::Uuid> = row.get(i);
                match uuid_ {
                    Some(u) => Data::from(Some(u.to_string())),
                    None => Data::Null,
                }
            }
            "inet" => {
                let ip: Option<IpAddr> = row.get(i);
                match ip {
                    Some(i) => Data::from(Some(i.to_string())),
                    None => Data::Null,
                }
            }
            "numeric" => {
                let decimal: Option<Decimal> = row.get(i);
                match decimal {
                    Some(d) => Data::Decimal(d.to_f64().unwrap_or_else(|| f64::NAN)),
                    None => Data::Null,
                }
            }
            _ => unimplemented!("Unimplemented conversion for type: '{}'", type_.name()),
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
    ptr as _
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
        let con = Database::new(CONNECTION_URI);
        con.execute("create table if not exists foobar (col1 integer, col2 integer)");
        let n_rows = con.execute("insert into foobar (col1, col2) values (1, 1)");
        assert_eq!(n_rows, 1)
    }
}
