//#![warn(missing_docs)]
use postgres as pg;
use postgres::fallible_iterator::FallibleIterator;
use postgres::types::{FromSql, Type};
use postgres::RowIter;
use rust_decimal::prelude::{Decimal, ToPrimitive};
use std::error::Error;
use std::ffi::CString;
use std::net::IpAddr;
use std::os::raw::c_char;
use std::{ffi, mem};
use time;

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
    pub ptr: *mut u8,
    pub len: u32,
}

impl From<Vec<u8>> for BytesPtr {
    fn from(mut v: Vec<u8>) -> Self {
        v.shrink_to_fit();
        let ptr = v.as_ptr() as _;
        let len = v.len() as u32;
        mem::forget(v);
        BytesPtr { ptr, len }
    }
}

#[derive(Debug)]
#[repr(C)]
pub struct StringPtr {
    pub ptr: *mut c_char,
    pub len: u32,
}

impl From<CString> for StringPtr {
    fn from(cstring: CString) -> Self {
        let len = cstring.as_bytes().len() as _;
        let ptr = cstring.into_raw() as _;
        StringPtr { ptr, len }
    }
}

#[derive(Debug)]
#[repr(C)]
pub struct DateInfo {
    /// The value represents the number of days since January 1st, 2000.
    offset: i32,
    ptr: *const i32,
}

impl FromSql<'_> for DateInfo {
    fn from_sql(_: &Type, raw: &[u8]) -> std::result::Result<Self, Box<dyn Error + Sync + Send>> {
        let offset = postgres_protocol::types::date_from_sql(raw)?;
        let ptr = &offset as *const _;
        Ok(Self { offset, ptr })
    }
    fn accepts(ty: &Type) -> bool {
        ty == &Type::DATE
    }
}

#[derive(Debug)]
#[repr(C)]
pub struct TimeInfo {
    hour: u8,
    minute: u8,
    second: u8,
    usecond: u32,
}

impl FromSql<'_> for TimeInfo {
    fn from_sql(ty: &Type, raw: &[u8]) -> std::result::Result<Self, Box<dyn Error + Sync + Send>> {
        let t = time::Time::from_sql(ty, raw)?;
        let (hour, minute, second, usecond) = t.as_hms_micro();
        Ok(Self {
            hour,
            minute,
            second,
            usecond,
        })
    }
    fn accepts(ty: &Type) -> bool {
        ty == &Type::TIME
    }
}

#[derive(Debug)]
#[repr(C)]
pub struct DateTimeInfo {
    /// The value represents the number of microseconds since midnight, January 1st, 2000.
    offset: i64, // holds actual value, garbage value if offset is null ptr
    ptr: *const i64, // signals if the value from db was null
}
impl FromSql<'_> for DateTimeInfo {
    fn from_sql(_: &Type, raw: &[u8]) -> std::result::Result<Self, Box<dyn Error + Sync + Send>> {
        let offset = postgres_protocol::types::timestamp_from_sql(raw)?;
        let ptr = &offset as *const _;
        Ok(Self { offset, ptr })
    }
    fn accepts(ty: &Type) -> bool {
        ty == &Type::TIMESTAMP || ty == &Type::TIMESTAMPTZ
    }
}

#[derive(Debug)]
#[repr(C)]
pub enum Data {
    Bytes(BytesPtr),
    Date(DateInfo),
    DateTime(DateTimeInfo),
    Time(TimeInfo),
    Boolean(bool),
    Decimal(f64), // TODO: support lossless decimal/numeric type handling
    Int8(i8),
    Int16(i16),
    Int32(i32),
    Uint32(u32),
    Int64(i64),
    Float32(f32),
    Float64(f64),
    String(StringPtr),
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
simple_from!(f64, Float64);
simple_from!(f32, Float32);
simple_from!(DateInfo, Date);
simple_from!(DateTimeInfo, DateTime);
simple_from!(TimeInfo, Time);
simple_from!(BytesPtr, Bytes);
simple_from!(StringPtr, String);

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
    let columns = row.columns();
    for i in 0..row.len() {
        let type_ = unsafe { columns.get_unchecked(i).type_() };
        // TODO: postgres-types: expose Inner enum which these variations
        // and have postgres Row.type/(or something) expose the variant
        let value: Data = match type_.name() {
            "bytea" => row.get::<_, Option<Vec<u8>>>(i).map(BytesPtr::from).into(),
            "char" => row.get::<_, Option<i8>>(i).into(),
            "smallint" | "smallserial" | "int2" => row.get::<_, Option<i16>>(i).into(),
            "oid" => row.get::<_, Option<u32>>(i).into(),
            "int4" | "int" | "serial" => row.get::<_, Option<i32>>(i).into(),
            "bigint" | "int8" | "bigserial" => row.get::<_, Option<i64>>(i).into(),
            "bool" => row.get::<_, Option<bool>>(i).into(),
            "double precision" | "float8" => row
                .get::<_, Option<f64>>(i)
                .or_else(|| Some(f64::NAN))
                .into(),
            "real" | "float4" => row
                .get::<_, Option<f32>>(i)
                .or_else(|| Some(f32::NAN))
                .into(),
            "varchar" | "char(n)" | "text" | "citext" | "name" | "unknown" | "bpchar" => row
                .get::<_, Option<String>>(i)
                .map(|v| CString::new(v).unwrap())
                .map(StringPtr::from)
                .into(),
            "timestamp" | "timestamp with time zone" | "timestamptz" => row
                .get::<_, Option<DateTimeInfo>>(i)
                .or_else(|| {
                    Some(DateTimeInfo {
                        offset: 0, // garbage value
                        ptr: std::ptr::null(),
                    })
                })
                .into(),
            "date" => row
                .get::<_, Option<DateInfo>>(i)
                .or_else(|| {
                    Some(DateInfo {
                        offset: 0, // garbage value
                        ptr: std::ptr::null(),
                    })
                })
                .into(),
            "time" => row.get::<_, Option<TimeInfo>>(i).into(),
            "json" | "jsonb" => row
                .get::<_, Option<serde_json::Value>>(i)
                .map(|v| CString::new(v.to_string()).unwrap())
                .map(StringPtr::from)
                .into(),
            "uuid" => row
                .get::<_, Option<uuid::Uuid>>(i)
                .map(|u| CString::new(u.to_string()).unwrap())
                .map(StringPtr::from)
                .into(),
            "inet" => row
                .get::<_, Option<IpAddr>>(i)
                .map(|i| CString::new(i.to_string()).unwrap())
                .map(StringPtr::from)
                .into(),
            "numeric" => row
                .get::<_, Option<Decimal>>(i)
                .map(|v| v.to_f64().unwrap_or_else(|| f64::NAN))
                .or_else(|| Some(f64::NAN))
                .into(),
            _ => {
                let msg = format!(
                    "Unimplemented conversion for: '{}'; consider casting to text",
                    type_.name()
                );
                mem::forget(values);
                return Err(msg.into());
            }
        };
        *unsafe { values.get_unchecked_mut(i) } = value;
    }
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
