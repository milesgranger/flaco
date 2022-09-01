//#![warn(missing_docs)]
use arrow2::{
    array,
    array::{
        Array, MutableArray, MutableBinaryArray, MutableBooleanArray, MutableFixedSizeBinaryArray,
        MutablePrimitiveArray,
    },
    datatypes::{DataType, TimeUnit},
};
use postgres as pg;
use postgres::fallible_iterator::FallibleIterator;
use postgres::types::Type;
use std::iter::Iterator;
use std::{any::Any, collections::BTreeMap};
use time;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub type Table = BTreeMap<String, Column>;

pub struct Column {
    array: Box<dyn Array>,
    dtype: DataType,
}
impl Column {
    pub fn new(array: impl MutableArray) -> Self {
        let mut array = array;
        Self {
            dtype: array.data_type().clone(),
            array: array.as_box(),
        }
    }
    pub fn dtype(&self) -> &DataType {
        &self.dtype
    }
    pub fn inner_mut<T: Any + 'static>(&mut self) -> &mut T {
        self.array.as_any_mut().downcast_mut::<T>().unwrap()
    }
    pub fn push<V, T: array::TryPush<V> + Any + 'static>(&mut self, value: V) -> Result<()> {
        self.inner_mut::<T>().try_push(value)?;
        Ok(())
    }
}

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
                    .push::<_, MutableBooleanArray>(row.try_get(idx).ok())?;
            }
            &Type::CHAR => {
                table
                    .entry(column_name)
                    .or_insert_with(|| Column::new(MutablePrimitiveArray::<i8>::new()))
                    .push::<_, MutablePrimitiveArray<i8>>(row.try_get(idx).ok())?;
            }
            &Type::TEXT | &Type::VARCHAR | &Type::UNKNOWN | &Type::NAME => {
                table
                    .entry(column_name)
                    .or_insert_with(|| Column::new(MutableBinaryArray::<i32>::new()))
                    .push::<_, MutableBinaryArray<i32>>(row.try_get::<_, Vec<u8>>(idx).ok())?;
            }
            &Type::INT2 => {
                table
                    .entry(column_name)
                    .or_insert_with(|| Column::new(MutablePrimitiveArray::<i16>::new()))
                    .push::<_, MutablePrimitiveArray<i16>>(row.try_get(idx).ok())?;
            }
            &Type::INT4 => {
                table
                    .entry(column_name)
                    .or_insert_with(|| Column::new(MutablePrimitiveArray::<i32>::new()))
                    .push::<_, MutablePrimitiveArray<i32>>(row.try_get(idx).ok())?;
            }
            &Type::INT8 => {
                table
                    .entry(column_name)
                    .or_insert_with(|| Column::new(MutablePrimitiveArray::<i64>::new()))
                    .push::<_, MutablePrimitiveArray<i64>>(row.try_get(idx).ok())?;
            }
            &Type::FLOAT4 => {
                table
                    .entry(column_name)
                    .or_insert_with(|| Column::new(MutablePrimitiveArray::<f32>::new()))
                    .push::<_, MutablePrimitiveArray<f32>>(row.try_get(idx).ok())?;
            }
            &Type::FLOAT8 => {
                table
                    .entry(column_name)
                    .or_insert_with(|| Column::new(MutablePrimitiveArray::<f64>::new()))
                    .push::<_, MutablePrimitiveArray<f64>>(row.try_get(idx).ok())?;
            }
            &Type::TIMESTAMP => {
                table
                    .entry(column_name)
                    .or_insert_with(|| {
                        Column::new(
                            MutablePrimitiveArray::<i64>::new()
                                .to(DataType::Time64(TimeUnit::Microsecond)),
                        )
                    })
                    .push::<_, MutablePrimitiveArray<i64>>(row.try_get(idx).ok())?;
            }
            &Type::TIMESTAMPTZ => {
                let offset = row
                    .try_get::<_, time::OffsetDateTime>(idx)
                    .ok()
                    .map(|v| v.offset().to_string());
                table
                    .entry(column_name)
                    .or_insert_with(|| {
                        Column::new(
                            MutablePrimitiveArray::<i64>::new()
                                .to(DataType::Timestamp(TimeUnit::Microsecond, offset)),
                        )
                    })
                    .push::<_, MutablePrimitiveArray<i64>>(row.try_get(idx).ok())?;
            }
            &Type::DATE => {
                table
                    .entry(column_name)
                    .or_insert_with(|| {
                        Column::new(MutablePrimitiveArray::<i32>::new().to(DataType::Date32))
                    })
                    .push::<_, MutablePrimitiveArray<i32>>(row.try_get(idx).ok())?;
            }
            &Type::TIME => {
                table
                    .entry(column_name)
                    .or_insert_with(|| {
                        Column::new(
                            MutablePrimitiveArray::<i64>::new()
                                .to(DataType::Time64(TimeUnit::Microsecond)),
                        )
                    })
                    .push::<_, MutablePrimitiveArray<i64>>(row.try_get(idx).ok())?;
            }
            &Type::TIMETZ => {
                // TIMETZ is 12 bytes; Fixed size binary array then since no DataType matches
                table
                    .entry(column_name)
                    .or_insert_with(|| Column::new(MutableFixedSizeBinaryArray::new(12)))
                    .inner_mut::<MutableFixedSizeBinaryArray>()
                    .push(row.try_get::<_, Vec<u8>>(idx).ok());
            }
            &Type::INTERVAL => {
                // INTERVAL is 16 bytes; Fixed size binary array then sinece i128 not impl FromSql
                table
                    .entry(column_name)
                    .or_insert_with(|| Column::new(MutableFixedSizeBinaryArray::new(16)))
                    .inner_mut::<MutableFixedSizeBinaryArray>()
                    .push(row.try_get::<_, Vec<u8>>(idx).ok());
            }
            _ => unimplemented!(
                "Type {} not implemented, consider opening an issue or casting to text.",
                column.type_()
            ),
        }
    }
    Ok(())
}
