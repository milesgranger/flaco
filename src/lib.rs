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
use postgres::types::{FromSql, Type};
use postgres::RowIter;
use rust_decimal::prelude::{Decimal, ToPrimitive};
use std::collections::HashMap;
use std::error::Error;
use std::iter::Iterator;
use std::net::IpAddr;
use std::{any::Any, collections::BTreeMap};
use time;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub type Table = BTreeMap<String, Column>;

pub struct Column {
    array: Box<dyn Array>,
    dtype: DataType,
    name: String,
}
impl Column {
    pub fn new(array: impl MutableArray, name: impl ToString) -> Self {
        let mut array = array;
        Self {
            dtype: array.data_type().clone(),
            array: array.as_box(),
            name: name.to_string(),
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
        append_row(&mut table, &row);
    }
    Ok(table)
}

#[inline(always)]
fn append_row(table: &mut BTreeMap<String, Column>, row: &pg::Row) -> Result<()> {
    for (idx, row_column) in row.columns().iter().enumerate() {
        let column_name = row_column.name().to_string();
        match row_column.type_() {
            &Type::BYTEA => {
                table
                    .entry(column_name)
                    .or_insert_with(|| {
                        Column::new(MutableBinaryArray::<i32>::new(), row_column.name())
                    })
                    .push::<_, MutableBinaryArray<i32>>(row.get::<_, Option<Vec<u8>>>(idx))?;
            }
            &Type::BOOL => {
                table
                    .entry(column_name)
                    .or_insert_with(|| Column::new(MutableBooleanArray::new(), row_column.name()))
                    .push::<_, MutableBooleanArray>(row.try_get(idx).ok())?;
            }
            &Type::CHAR => {
                table
                    .entry(column_name)
                    .or_insert_with(|| {
                        Column::new(MutablePrimitiveArray::<i8>::new(), row_column.name())
                    })
                    .push::<_, MutablePrimitiveArray<i8>>(row.try_get(idx).ok())?;
            }
            &Type::TEXT | &Type::VARCHAR | &Type::UNKNOWN | &Type::NAME => {
                table
                    .entry(column_name)
                    .or_insert_with(|| {
                        Column::new(MutableBinaryArray::<i32>::new(), row_column.name())
                    })
                    .push::<_, MutableBinaryArray<i32>>(row.try_get::<_, Vec<u8>>(idx).ok())?;
            }
            &Type::INT2 => {
                table
                    .entry(column_name)
                    .or_insert_with(|| {
                        Column::new(MutablePrimitiveArray::<i16>::new(), row_column.name())
                    })
                    .push::<_, MutablePrimitiveArray<i16>>(row.try_get(idx).ok())?;
            }
            &Type::INT4 => {
                table
                    .entry(column_name)
                    .or_insert_with(|| {
                        Column::new(MutablePrimitiveArray::<i32>::new(), row_column.name())
                    })
                    .push::<_, MutablePrimitiveArray<i32>>(row.try_get(idx).ok())?;
            }
            &Type::INT8 => {
                table
                    .entry(column_name)
                    .or_insert_with(|| {
                        Column::new(MutablePrimitiveArray::<i64>::new(), row_column.name())
                    })
                    .push::<_, MutablePrimitiveArray<i64>>(row.try_get(idx).ok())?;
            }
            &Type::FLOAT4 => {
                table
                    .entry(column_name)
                    .or_insert_with(|| {
                        Column::new(MutablePrimitiveArray::<f32>::new(), row_column.name())
                    })
                    .push::<_, MutablePrimitiveArray<f32>>(row.try_get(idx).ok())?;
            }
            &Type::FLOAT8 => {
                table
                    .entry(column_name)
                    .or_insert_with(|| {
                        Column::new(MutablePrimitiveArray::<f64>::new(), row_column.name())
                    })
                    .push::<_, MutablePrimitiveArray<f64>>(row.try_get(idx).ok())?;
            }
            &Type::TIMESTAMP => {
                table
                    .entry(column_name)
                    .or_insert_with(|| {
                        Column::new(
                            MutablePrimitiveArray::<i64>::new()
                                .to(DataType::Time64(TimeUnit::Microsecond)),
                            row_column.name(),
                        )
                    })
                    .push::<_, MutablePrimitiveArray<i64>>(row.try_get(idx).ok())?;
            }
            _ => todo!(),
        }
    }
    Ok(())
}
