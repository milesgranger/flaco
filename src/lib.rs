//#![warn(missing_docs)]
use postgres as pg;
use postgres::fallible_iterator::FallibleIterator;
use postgres::RowIter;
use rust_decimal::prelude::{Decimal, ToPrimitive};
use std::ffi::CString;
use std::net::IpAddr;
use std::os::raw::c_char;
use std::{ffi, mem};
use time;
use time::format_description::well_known::Rfc3339;

type DatabasePtr = *mut u32;
type RowIteratorPtr = *mut u32;
type RowDataArrayPtr = *mut u32;
type RowColumnNamesArrayPtr = *mut *mut c_char;
type Exception = *mut c_char;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

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

    pub fn connect(&mut self) -> Result<()> {
        if self.client.is_none() {
            let con = pg::Client::connect(&self.uri, pg::NoTls)?;
            self.client = Some(con);
        }
        Ok(())
    }

    pub fn disconnect(&mut self) {
        if self.client.is_some() {
            self.client = None;
        }
    }
}

#[inline(always)]
fn string_into_exception<S: ToString>(msg: S, exc: &mut Exception) {
    let msg = CString::new(msg.to_string()).unwrap();
    *&mut *exc = msg.into_raw();
}

#[no_mangle]
pub extern "C" fn db_create(uri_ptr: *const c_char) -> DatabasePtr {
    let uri_c = unsafe { ffi::CStr::from_ptr(uri_ptr) };
    let uri = uri_c.to_str().unwrap();
    let db = Box::new(Database::new(uri));
    Box::into_raw(db) as *mut _
}

#[no_mangle]
pub extern "C" fn read_sql(
    stmt_ptr: *const c_char,
    db_ptr: DatabasePtr,
    exc: &mut Exception,
) -> RowIteratorPtr {
    let mut db = unsafe { Box::from_raw(db_ptr as *mut Database) };
    let stmt_c = unsafe { ffi::CStr::from_ptr(stmt_ptr) };
    let stmt = stmt_c.to_str().unwrap();
    let res = match db.client.as_mut() {
        Some(con) => match con.query_raw::<_, &i32, _>(stmt, &[]) {
            Ok(row_iter) => {
                let boxed_row_iter = Box::new(row_iter);
                Box::into_raw(boxed_row_iter) as RowIteratorPtr
            }
            Err(e) => {
                string_into_exception(e, exc);
                std::ptr::null_mut()
            }
        },
        None => {
            let msg = "Not connected. Use 'with Database(...) as con', or call '.connect()'";
            string_into_exception(msg, exc);
            std::ptr::null_mut()
        }
    };
    mem::forget(db);
    res
}

#[no_mangle]
pub extern "C" fn db_disconnect(ptr: DatabasePtr) {
    let mut conn = unsafe { Box::from_raw(ptr as *mut Database) };
    conn.disconnect();
    mem::forget(conn);
}

#[no_mangle]
pub extern "C" fn db_connect(ptr: DatabasePtr, exc: &mut Exception) {
    let mut db = unsafe { Box::from_raw(ptr as *mut Database) };
    if let Err(err) = db.connect() {
        string_into_exception(err, exc);
    };
    mem::forget(db);
}

#[derive(Debug)]
#[repr(C)]
pub struct BytesPtr {
    ptr: *mut u8,
    len: u32,
}

#[derive(Debug)]
#[repr(C)]
pub enum Data {
    Bytes(BytesPtr),
    Boolean(bool),
    Decimal(f64), // TODO: support lossless decimal/numeric type handling
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
pub extern "C" fn free_db(ptr: DatabasePtr) {
    unsafe { Box::from_raw(ptr as DatabasePtr) };
}

fn free_row_iter(ptr: &mut RowIteratorPtr) {
    let _ = unsafe { Box::from_raw(*ptr as *mut pg::RowIter) };
    *ptr = std::ptr::null_mut();
}

#[no_mangle]
pub extern "C" fn next_row(
    row_iter_ptr: &mut RowIteratorPtr,
    row_data_array_ptr: &mut RowDataArrayPtr,
    n_columns: &mut u32,
    column_names: &mut RowColumnNamesArrayPtr,
    exc: &mut Exception,
) {
    let mut row_iter = unsafe { Box::from_raw(*row_iter_ptr as *mut RowIter) };
    match row_iter.next() {
        Ok(maybe_row) => match maybe_row {
            Some(row) => {
                if row_data_array_ptr.is_null() {
                    *&mut *row_data_array_ptr = init_row_data_array(&row) as _;
                }
                if column_names.is_null() {
                    *&mut *column_names = row_column_names(&row) as _;
                    *n_columns = row.len() as _;
                }
                if let Err(err) = row_data(row, row_data_array_ptr) {
                    string_into_exception(err, exc);
                    let len = *n_columns;
                    free_row_data_array(row_data_array_ptr, len);
                    free_row_column_names(column_names, len as usize);
                    free_row_iter(row_iter_ptr);
                };
            }
            None => {
                let len = *n_columns;
                free_row_data_array(row_data_array_ptr, len);
                free_row_column_names(column_names, len as usize);
                free_row_iter(row_iter_ptr);
            }
        },
        Err(err) => {
            string_into_exception(err, exc);
            let len = *n_columns;
            free_row_data_array(row_data_array_ptr, len);
            free_row_column_names(column_names, len as usize);
            free_row_iter(row_iter_ptr);
        }
    };
    mem::forget(row_iter);
}

pub fn init_row_data_array(row: &pg::Row) -> RowDataArrayPtr {
    let len = row.len();
    let mut values = Vec::with_capacity(row.len());
    for _ in 0..len {
        values.push(Data::Null);
    }
    values.shrink_to_fit();
    let ptr = values.as_mut_ptr();
    mem::forget(values);
    ptr as _
}

fn row_data(row: pg::Row, array_ptr: &mut RowDataArrayPtr) -> Result<()> {
    let mut values = unsafe { Vec::from_raw_parts(*array_ptr as _, row.len(), row.len()) };
    values.clear();
    for i in 0..row.len() {
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
            "real" | "float4" => {
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
            _ => {
                let msg = format!(
                    "Unimplemented conversion for: '{}'; consider casting to text",
                    type_.name()
                );
                mem::forget(values);
                return Err(msg.into());
            }
        };
        values.push(val);
    }
    //assert_eq!(values.len(), values.capacity());
    mem::forget(values);
    Ok(())
}

fn free_row_data_array(ptr: &mut RowDataArrayPtr, len: u32) {
    let _: Vec<Data> = unsafe { Vec::from_raw_parts(*ptr as _, len as usize, len as usize) };
    *&mut *ptr = std::ptr::null_mut() as RowDataArrayPtr as _;
}

fn row_column_names(row: &pg::Row) -> RowColumnNamesArrayPtr {
    let mut names = row
        .columns()
        .iter()
        .map(|col| col.name())
        .map(|name| ffi::CString::new(name).unwrap())
        .map(|name| name.into_raw() as _)
        .collect::<Vec<*const c_char>>();
    names.shrink_to_fit();
    let ptr = names.as_ptr();
    mem::forget(names);
    ptr as _
}

fn free_row_column_names(ptr: &mut RowColumnNamesArrayPtr, len: usize) {
    let _names = unsafe { Vec::from_raw_parts(*ptr, len, len) };
    *ptr = std::ptr::null_mut();
}

#[no_mangle]
pub extern "C" fn index_row(row_data_array_ptr: RowDataArrayPtr, len: u32, idx: u32) -> *mut Data {
    let mut row: Vec<Data> =
        unsafe { Vec::from_raw_parts(row_data_array_ptr as _, len as usize, len as usize) };
    let data = &mut row[idx as usize] as *mut Data;
    mem::forget(row);
    data
}
