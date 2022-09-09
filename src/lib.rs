use arrow2::array::{
    Array, BinaryArray, MutableBinaryArray, MutableBooleanArray, MutableFixedSizeBinaryArray,
    MutablePrimitiveArray, MutableUtf8Array, Utf8Array,
};
use arrow2::chunk::Chunk;
use arrow2::datatypes::{DataType, Schema};
use arrow2::io::{ipc, parquet};
use arrow2::{array, array::MutableArray};
use numpy::IntoPyArray;
use pyo3::create_exception;
use pyo3::exceptions::PyException;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::any::Any;
use std::collections::BTreeMap;
use std::fs::File;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

create_exception!(flaco, FlacoException, PyException);

#[pymodule]
fn flaco(py: Python, m: &PyModule) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(read_sql_to_file, m)?)?;
    m.add_function(wrap_pyfunction!(read_sql_to_numpy, m)?)?;
    m.add_class::<FileFormat>()?;
    m.add("FlacoException", py.get_type::<FlacoException>())?;
    Ok(())
}

#[pyclass(name = "FileFormat")]
#[derive(Clone, Debug)]
pub enum FileFormat {
    Feather,
    Parquet,
}

#[inline(always)]
fn to_py_err(err: impl ToString) -> PyErr {
    PyErr::new::<FlacoException, _>(err.to_string())
}

// TODO: Stream data into a file in chunks during query reading
#[pyfunction]
pub fn read_sql_to_file(uri: &str, stmt: &str, path: &str, format: FileFormat) -> PyResult<()> {
    let mut client = postgres::Client::connect(uri, postgres::NoTls).map_err(to_py_err)?;
    let table = postgresql::read_sql(&mut client, stmt).map_err(to_py_err)?;
    match format {
        FileFormat::Feather => write_table_to_feather(table, path).map_err(to_py_err)?,
        FileFormat::Parquet => write_table_to_parquet(table, path).map_err(to_py_err)?,
    }
    Ok(())
}

#[pyfunction]
pub fn read_sql_to_numpy<'a, 'py>(
    py: Python<'py>,
    uri: &'a str,
    stmt: &'a str,
) -> PyResult<BTreeMap<String, PyObject>> {
    let mut client = postgres::Client::connect(uri, postgres::NoTls).map_err(to_py_err)?;
    let table = postgresql::read_sql(&mut client, stmt).map_err(to_py_err)?;
    let mut result = BTreeMap::new();
    for (name, column) in table {
        result.insert(name, column.into_pyarray(py));
    }
    Ok(result)
}

pub type Table = BTreeMap<String, Column>;

pub struct Column {
    pub array: Box<dyn MutableArray>,
    pub dtype: DataType,
}

impl Column {
    pub fn new(array: impl MutableArray + 'static) -> Self {
        Self {
            dtype: array.data_type().clone(),
            array: Box::new(array),
        }
    }
    pub fn dtype(&self) -> &DataType {
        &self.dtype
    }
    pub fn inner_mut<T: Any + 'static>(&mut self) -> &mut T {
        self.array.as_mut_any().downcast_mut::<T>().unwrap()
    }
    pub fn inner<T: Any + 'static>(&self) -> &T {
        self.array.as_any().downcast_ref::<T>().unwrap()
    }
    pub fn push<V, T: array::TryPush<V> + Any + 'static>(&mut self, value: V) -> Result<()> {
        self.inner_mut::<T>().try_push(value)?;
        Ok(())
    }
    fn contains_nulls(&self) -> bool {
        self.array
            .validity()
            .map(|v| v.unset_bits() > 0)
            .unwrap_or_else(|| false)
    }
    fn is_float(&self) -> bool {
        [DataType::Float16, DataType::Float32, DataType::Float64].contains(&self.dtype)
    }
    pub fn into_pyarray(mut self, py: Python) -> PyObject {
        match self.dtype {
            DataType::Boolean => self
                .inner::<MutableBooleanArray>()
                .iter()
                .map(|v| v.to_object(py))
                .collect::<Vec<_>>()
                .into_pyarray(py)
                .to_object(py),
            DataType::Binary => self
                .inner_mut::<MutableBinaryArray<i32>>()
                .as_arc()
                .as_ref()
                .as_any()
                .downcast_ref::<BinaryArray<i32>>()
                .unwrap()
                .iter()
                .map(|v| v.to_object(py))
                .collect::<Vec<PyObject>>()
                .into_pyarray(py)
                .to_object(py),

            DataType::Utf8 => self
                .inner_mut::<MutableUtf8Array<i32>>()
                .as_arc()
                .as_ref()
                .as_any()
                .downcast_ref::<Utf8Array<i32>>()
                .unwrap()
                .iter()
                .map(|v| v.to_object(py))
                .collect::<Vec<PyObject>>()
                .into_pyarray(py)
                .to_object(py),
            DataType::Int8 => self
                .inner::<MutablePrimitiveArray<i8>>()
                .iter()
                .map(|v| v.to_object(py))
                .collect::<Vec<PyObject>>()
                .into_pyarray(py)
                .to_object(py),
            DataType::Int16 => self
                .inner::<MutablePrimitiveArray<i16>>()
                .iter()
                .map(|v| v.to_object(py))
                .collect::<Vec<PyObject>>()
                .into_pyarray(py)
                .to_object(py),
            DataType::Int32 => self
                .inner::<MutablePrimitiveArray<i32>>()
                .iter()
                .map(|v| v.to_object(py))
                .collect::<Vec<PyObject>>()
                .into_pyarray(py)
                .to_object(py),
            DataType::UInt32 => self
                .inner::<MutablePrimitiveArray<u32>>()
                .iter()
                .map(|v| v.to_object(py))
                .collect::<Vec<PyObject>>()
                .into_pyarray(py)
                .to_object(py),
            DataType::Int64 => self
                .inner::<MutablePrimitiveArray<i64>>()
                .iter()
                .map(|v| v.to_object(py))
                .collect::<Vec<PyObject>>()
                .into_pyarray(py)
                .to_object(py),
            DataType::UInt64 => self
                .inner::<MutablePrimitiveArray<u64>>()
                .iter()
                .map(|v| v.to_object(py))
                .collect::<Vec<PyObject>>()
                .into_pyarray(py)
                .to_object(py),
            DataType::Float32 => self
                .inner::<MutablePrimitiveArray<f32>>()
                .iter()
                .map(|v| v.to_object(py))
                .collect::<Vec<PyObject>>()
                .into_pyarray(py)
                .to_object(py),
            DataType::Float64 => self
                .inner::<MutablePrimitiveArray<f64>>()
                .iter()
                .map(|v| v.to_object(py))
                .collect::<Vec<PyObject>>()
                .into_pyarray(py)
                .to_object(py),
            DataType::FixedSizeBinary(_) => self
                .inner::<MutableFixedSizeBinaryArray>()
                .iter()
                .map(|v| v.to_object(py))
                .collect::<Vec<PyObject>>()
                .into_pyarray(py)
                .to_object(py),
            _ => unimplemented!(
                "Dtype: {:?} not implemented for conversion to numpy",
                &self.dtype
            ),
        }
    }
}

fn write_table_to_parquet(table: Table, path: &str) -> Result<()> {
    let mut fields = vec![];
    let mut arrays = vec![];
    for (name, mut column) in table.into_iter() {
        fields.push(arrow2::datatypes::Field::new(name, column.dtype, true));
        arrays.push(column.array.as_box());
    }
    let schema = Schema::from(fields);
    let chunks = Chunk::new(arrays);
    let options = parquet::write::WriteOptions {
        write_statistics: true,
        compression: parquet::write::CompressionOptions::Uncompressed,
        version: parquet::write::Version::V2,
    };
    let encodings = schema
        .fields
        .iter()
        .map(|f| parquet::write::transverse(&f.data_type, |_| parquet::write::Encoding::Plain))
        .collect();
    let row_groups = parquet::write::RowGroupIterator::try_new(
        vec![Ok(chunks)].into_iter(),
        &schema,
        options,
        encodings,
    )?;
    let file = File::create(path)?;
    let mut writer = parquet::write::FileWriter::try_new(file, schema, options)?;
    for group in row_groups {
        writer.write(group?)?;
    }
    writer.end(None)?;
    Ok(())
}

fn write_table_to_feather(table: Table, path: &str) -> Result<()> {
    let mut fields = vec![];
    let mut arrays = vec![];
    for (name, mut column) in table.into_iter() {
        fields.push(arrow2::datatypes::Field::new(name, column.dtype, true));
        arrays.push(column.array.as_box());
    }
    let schema = Schema::from(fields);
    let options = ipc::write::WriteOptions { compression: None };

    let file = File::create(path.to_string())?;
    let mut writer = ipc::write::FileWriter::try_new(file, &schema, None, options)?;

    let chunk = Chunk::try_new(arrays)?;
    writer.write(&chunk, None)?;
    writer.finish()?;

    Ok(())
}

pub mod postgresql {
    use super::{Column, Result, Table};
    use arrow2::array::{
        MutableBinaryArray, MutableBooleanArray, MutableFixedSizeBinaryArray,
        MutablePrimitiveArray, MutableUtf8Array,
    };

    use postgres as pg;
    use postgres::fallible_iterator::FallibleIterator;
    use postgres::types::Type;
    use rust_decimal::{prelude::ToPrimitive, Decimal};
    use std::collections::BTreeMap;
    use std::{iter::Iterator, net::IpAddr};
    use time;

    pub fn read_sql(client: &mut pg::Client, sql: &str) -> Result<Table> {
        let mut row_iter = client.query_raw::<_, &i32, _>(sql, &[])?;
        let mut table = BTreeMap::new();
        while let Some(row) = row_iter.next()? {
            append_row(&mut table, &row)?;
        }
        Ok(table)
    }

    #[inline(always)]
    fn append_row(table: &mut BTreeMap<String, Column>, row: &pg::Row) -> Result<()> {
        for (idx, column) in row.columns().iter().enumerate() {
            let column_name = column.name().to_string();
            match column.type_() {
                &Type::BYTEA => {
                    table
                        .entry(column_name)
                        .or_insert_with(|| Column::new(MutableBinaryArray::<i32>::new()))
                        .push::<_, MutableBinaryArray<i32>>(row.get::<_, Option<Vec<u8>>>(idx))?;
                }
                &Type::BOOL => {
                    table
                        .entry(column_name)
                        .or_insert_with(|| Column::new(MutableBooleanArray::new()))
                        .push::<_, MutableBooleanArray>(row.get::<_, Option<bool>>(idx))?;
                }
                &Type::CHAR => {
                    table
                        .entry(column_name)
                        .or_insert_with(|| Column::new(MutablePrimitiveArray::<i8>::new()))
                        .push::<_, MutablePrimitiveArray<i8>>(row.get::<_, Option<i8>>(idx))?;
                }
                &Type::TEXT | &Type::VARCHAR | &Type::UNKNOWN | &Type::NAME | &Type::BPCHAR => {
                    table
                        .entry(column_name)
                        .or_insert_with(|| Column::new(MutableUtf8Array::<i32>::new()))
                        .push::<_, MutableUtf8Array<i32>>(row.get::<_, Option<String>>(idx))?;
                }
                &Type::JSON | &Type::JSONB => {
                    table
                        .entry(column_name)
                        .or_insert_with(|| Column::new(MutableUtf8Array::<i32>::new()))
                        .push::<_, MutableUtf8Array<i32>>(
                            row.get::<_, Option<serde_json::Value>>(idx)
                                .map(|v| v.to_string()),
                        )?;
                }
                &Type::OID => {
                    table
                        .entry(column_name)
                        .or_insert_with(|| Column::new(MutablePrimitiveArray::<u32>::new()))
                        .push::<_, MutablePrimitiveArray<u32>>(row.get::<_, Option<u32>>(idx))?;
                }
                &Type::UUID => {
                    table
                        .entry(column_name)
                        .or_insert_with(|| Column::new(MutableUtf8Array::<i32>::new()))
                        .push::<_, MutableUtf8Array<i32>>(
                            row.get::<_, Option<IpAddr>>(idx).map(|v| v.to_string()),
                        )?;
                }
                &Type::INT2 => {
                    table
                        .entry(column_name)
                        .or_insert_with(|| Column::new(MutablePrimitiveArray::<i16>::new()))
                        .push::<_, MutablePrimitiveArray<i16>>(row.get::<_, Option<i16>>(idx))?;
                }
                &Type::INT4 => {
                    table
                        .entry(column_name)
                        .or_insert_with(|| Column::new(MutablePrimitiveArray::<i32>::new()))
                        .push::<_, MutablePrimitiveArray<i32>>(row.get::<_, Option<i32>>(idx))?;
                }
                &Type::INT8 => {
                    table
                        .entry(column_name)
                        .or_insert_with(|| Column::new(MutablePrimitiveArray::<i64>::new()))
                        .push::<_, MutablePrimitiveArray<i64>>(row.get::<_, Option<i64>>(idx))?;
                }
                &Type::FLOAT4 => {
                    table
                        .entry(column_name)
                        .or_insert_with(|| Column::new(MutablePrimitiveArray::<f32>::new()))
                        .push::<_, MutablePrimitiveArray<f32>>(row.get::<_, Option<f32>>(idx))?;
                }
                &Type::FLOAT8 => {
                    table
                        .entry(column_name)
                        .or_insert_with(|| Column::new(MutablePrimitiveArray::<f64>::new()))
                        .push::<_, MutablePrimitiveArray<f64>>(row.get::<_, Option<f64>>(idx))?;
                }
                &Type::TIMESTAMP => {
                    table
                        .entry(column_name)
                        .or_insert_with(|| Column::new(MutableUtf8Array::<i32>::new()))
                        .push::<_, MutableUtf8Array<i32>>(
                            row.get::<_, Option<time::PrimitiveDateTime>>(idx)
                                .map(|v| v.to_string()),
                        )?;
                }
                &Type::TIMESTAMPTZ => {
                    table
                        .entry(column_name)
                        .or_insert_with(|| Column::new(MutableUtf8Array::<i32>::new()))
                        .push::<_, MutableUtf8Array<i32>>(
                            row.get::<_, Option<time::OffsetDateTime>>(idx)
                                .map(|v| v.to_string()),
                        )?;
                }
                &Type::DATE => {
                    table
                        .entry(column_name)
                        .or_insert_with(|| Column::new(MutableUtf8Array::<i32>::new()))
                        .push::<_, MutableUtf8Array<i32>>(
                            row.get::<_, Option<time::Date>>(idx).map(|v| v.to_string()),
                        )?;
                }
                &Type::TIME => {
                    table
                        .entry(column_name)
                        .or_insert_with(|| Column::new(MutableUtf8Array::<i32>::new()))
                        .push::<_, MutableUtf8Array<i32>>(
                            row.get::<_, Option<time::Time>>(idx).map(|v| v.to_string()),
                        )?;
                }
                &Type::TIMETZ => {
                    // TIMETZ is 12 bytes; Fixed size binary array then since no DataType matches
                    table
                        .entry(column_name)
                        .or_insert_with(|| Column::new(MutableUtf8Array::<i32>::new()))
                        .push::<_, MutableUtf8Array<i32>>(
                            row.get::<_, Option<time::Time>>(idx).map(|v| v.to_string()),
                        )?;
                }
                &Type::INTERVAL => {
                    // INTERVAL is 16 bytes; Fixed size binary array then sinece i128 not impl FromSql
                    table
                        .entry(column_name)
                        .or_insert_with(|| Column::new(MutableFixedSizeBinaryArray::new(16)))
                        .inner_mut::<MutableFixedSizeBinaryArray>()
                        .push(row.get::<_, Option<Vec<u8>>>(idx));
                }
                &Type::NUMERIC => table
                    .entry(column_name)
                    .or_insert_with(|| Column::new(MutablePrimitiveArray::<f64>::new()))
                    .push::<_, MutablePrimitiveArray<f64>>(
                        row.get::<_, Option<Decimal>>(idx)
                            .map(|v| v.to_f64().unwrap_or_else(|| f64::NAN)),
                    )?,
                _ => unimplemented!(
                    "Type {} not implemented, consider opening an issue or casting to text.",
                    column.type_()
                ),
            }
        }
        Ok(())
    }
}
