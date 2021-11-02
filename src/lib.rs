//#![warn(missing_docs)]
use bumpalo::Bump;
use postgres as pg;
use postgres::fallible_iterator::FallibleIterator;
use postgres::RowIter;
use rust_decimal::prelude::{Decimal, ToPrimitive};
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
type SessionPtr = *mut u32;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;


#[no_mangle]
pub extern "C" fn init_session() -> SessionPtr {
    Box::into_raw(Box::new(Bump::new())) as SessionPtr
}

#[no_mangle]
pub extern "C" fn free_session(session: SessionPtr) {
    let mut bump = unsafe { Box::from_raw(session as *mut Bump) };
    bump.reset();
}

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
    fn from(v: Vec<u8>) -> Self {
        let mut v = v;
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

impl From<String> for StringPtr {
    fn from(v: String) -> Self {
        let cstring = CString::new(v).unwrap();
        let len = cstring.as_bytes().len() as _;

        let ptr = cstring.into_raw();
        StringPtr { ptr, len }
    }
}

#[derive(Debug)]
#[repr(C)]
pub struct DateInfo {
    year: i32,
    month: u8,
    day: u8,
}

#[derive(Debug)]
#[repr(C)]
pub struct TimeInfo {
    hour: u8,
    minute: u8,
    second: u8,
    usecond: u32,
}

#[derive(Debug)]
#[repr(C)]
pub struct DateTimeInfo {
    date: DateInfo,
    time: TimeInfo,
}

#[derive(Debug)]
#[repr(C)]
pub struct TzInfo {
    hours: i8,
    minutes: i8,
    seconds: i8,
    is_positive: bool,
}

#[derive(Debug)]
#[repr(C)]
pub struct DateTimeTzInfo {
    date: DateInfo,
    time: TimeInfo,
    tz: TzInfo,
}

#[derive(Debug)]
#[repr(C)]
pub enum Data {
    Bytes(*mut BytesPtr),
    Date(*mut DateInfo),
    DateTime(*mut DateTimeInfo),
    DateTimeTz(*mut DateTimeTzInfo),
    Time(*mut TimeInfo),
    Boolean(*mut bool),
    Decimal(*mut f64), // TODO: support lossless decimal/numeric type handling
    Int8(*mut i8),
    Int16(*mut i16),
    Int32(*mut i32),
    Uint32(*mut u32),
    Int64(*mut i64),
    Float32(*mut f32),
    Float64(*mut f64),
    String(*mut StringPtr),
    Null,
}

macro_rules! simple_from {
    ($t:ty, $i:ident) => {
        impl From<Option<*mut $t>> for Data {
            fn from(val: Option<*mut $t>) -> Self {
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
simple_from!(DateTimeTzInfo, DateTimeTz);
simple_from!(TimeInfo, Time);
simple_from!(BytesPtr, Bytes);
simple_from!(StringPtr, String);

#[inline(always)]
fn month_to_u8(month: time::Month) -> u8 {
    match month {
        time::Month::January => 1,
        time::Month::February => 2,
        time::Month::March => 3,
        time::Month::April => 4,
        time::Month::May => 5,
        time::Month::June => 6,
        time::Month::July => 7,
        time::Month::August => 8,
        time::Month::September => 9,
        time::Month::October => 10,
        time::Month::November => 11,
        time::Month::December => 12,
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
    session: &mut SessionPtr,
    exc: &mut Exception,
) {
    let mut row_iter = unsafe { Box::from_raw(*row_iter_ptr as *mut RowIter) };
    let mut bump = unsafe { Box::from_raw(*session as *mut Bump) };
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
                if let Err(err) = row_data(row, row_data_array_ptr, &mut bump) {
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
    mem::forget(bump);
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

fn row_data(row: pg::Row, array_ptr: &mut RowDataArrayPtr, bump: &mut Bump) -> Result<()> {
    let mut values = unsafe { Vec::from_raw_parts(*array_ptr as _, row.len(), row.len()) };
    assert_eq!(values.len(), values.capacity());
    assert_eq!(values.len(), row.len());
    bump.reset();
    for i in 0..row.len() {
        let type_ = row.columns()[i].type_();
        // TODO: postgres-types: expose Inner enum which these variations
        // and have postgres Row.type/(or something) expose the variant
        let value: Data = match type_.name() {
            "bytea" => row
                .get::<_, Option<Vec<u8>>>(i)
                .map(|v| bump.alloc(v.into()) as *mut BytesPtr)
                .into(),
            "char" => row
                .get::<_, Option<i8>>(i)
                .map(|v| bump.alloc(v) as *mut i8)
                .into(),
            "smallint" | "smallserial" | "int2" => row
                .get::<_, Option<i16>>(i)
                .map(|v| bump.alloc(v) as *mut i16)
                .into(),
            "oid" => row
                .get::<_, Option<u32>>(i)
                .map(|v| bump.alloc(v) as *mut u32)
                .into(),
            "int4" | "int" | "serial" => row
                .get::<_, Option<i32>>(i)
                .map(|v| bump.alloc(v) as *mut i32)
                .into(),
            "bigint" | "int8" | "bigserial" => row
                .get::<_, Option<i64>>(i)
                .map(|v| bump.alloc(v) as *mut i64)
                .into(),
            "bool" => row
                .get::<_, Option<bool>>(i)
                .map(|v| bump.alloc(v) as *mut bool)
                .into(),
            "double precision" | "float8" => row
                .get::<_, Option<f64>>(i)
                .or_else(|| Some(f64::NAN))
                .map(|v| bump.alloc(v) as *mut f64)
                .into(),
            "real" | "float4" => row
                .get::<_, Option<f32>>(i)
                .or_else(|| Some(f32::NAN))
                .map(|v| bump.alloc(v) as *mut f32)
                .into(),
            "varchar" | "char(n)" | "text" | "citext" | "name" | "unknown" | "bpchar" => row
                .get::<_, Option<String>>(i)
                .map(|v| bump.alloc(v.into()) as *mut StringPtr)
                .into(),
            "timestamp" => row
                .get::<_, Option<time::PrimitiveDateTime>>(i)
                .map(|t| {
                    bump.alloc(DateTimeInfo {
                        date: DateInfo {
                            year: t.year(),
                            month: month_to_u8(t.month()),
                            day: t.day(),
                        },
                        time: TimeInfo {
                            hour: t.hour(),
                            minute: t.minute(),
                            second: t.second(),
                            usecond: t.microsecond(),
                        },
                    }) as *mut DateTimeInfo
                })
                .into(),
            "timestamp with time zone" | "timestamptz" => row
                .get::<_, Option<time::OffsetDateTime>>(i)
                .map(|t| {
                    bump.alloc(DateTimeTzInfo {
                        date: DateInfo {
                            year: t.year() as _,
                            month: month_to_u8(t.month()),
                            day: t.day(),
                        },
                        time: TimeInfo {
                            hour: t.hour(),
                            minute: t.minute(),
                            second: t.second(),
                            usecond: t.microsecond(),
                        },
                        tz: TzInfo {
                            hours: t.offset().whole_hours(),
                            minutes: t.offset().minutes_past_hour(),
                            seconds: t.offset().seconds_past_minute(),
                            is_positive: t.offset().is_positive(),
                        },
                    }) as *mut DateTimeTzInfo
                })
                .into(),
            "date" => row
                .get::<_, Option<time::Date>>(i)
                .map(|t| {
                    bump.alloc(DateInfo {
                        year: t.year() as _,
                        month: month_to_u8(t.month()),
                        day: t.day(),
                    }) as *mut DateInfo
                })
                .into(),
            "time" => row
                .get::<_, Option<time::Time>>(i)
                .map(|t| {
                    bump.alloc(TimeInfo {
                        hour: t.hour(),
                        minute: t.minute(),
                        second: t.second(),
                        usecond: t.microsecond(),
                    }) as *mut TimeInfo
                })
                .into(),
            "json" | "jsonb" => row
                .get::<_, Option<serde_json::Value>>(i)
                .map(|v| bump.alloc(v.to_string().into()) as *mut StringPtr)
                .into(),
            "uuid" => row
                .get::<_, Option<uuid::Uuid>>(i)
                .map(|u| bump.alloc(u.to_string().into()) as *mut StringPtr)
                .into(),
            "inet" => row
                .get::<_, Option<IpAddr>>(i)
                .map(|i| bump.alloc(i.to_string().into()) as *mut StringPtr)
                .into(),
            "numeric" => row
                .get::<_, Option<Decimal>>(i)
                .map(|v| v.to_f64().unwrap_or_else(|| f64::NAN))
                .or_else(|| Some(f64::NAN))
                .map(|v| bump.alloc(v) as *mut f64)
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
        values[i] = value;
    }
    assert_eq!(values.len(), values.capacity());
    assert_eq!(values.len(), row.len());
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
