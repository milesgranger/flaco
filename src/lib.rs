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
use std::any::Any;
use std::collections::HashMap;
use std::error::Error;
use std::iter::Iterator;
use std::net::IpAddr;
use time;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub type Table = Vec<Column>;

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
    let mut table = if let Some(row) = row_iter.next()? {
        let mut table = init_table(&row)?;
        append_row(&mut table, &row)?;
        table
    } else {
        return Err("Query returned no rows!".into());
    };
    while let Some(row) = row_iter.next()? {
        append_row(&mut table, &row);
    }
    Ok(table)
}

#[inline(always)]
fn init_table(row: &pg::Row) -> Result<Vec<Column>> {
    let mut table = Vec::with_capacity(row.len());
    for column in row.columns() {
        let name = column.name();
        let col = match column.type_() {
            &Type::BYTEA => Column::new(MutableBinaryArray::<i32>::new(), name),
            &Type::BOOL => Column::new(MutableBooleanArray::new(), name),
            &Type::CHAR => Column::new(MutablePrimitiveArray::<i8>::new(), name),
            &Type::INT2 => Column::new(MutablePrimitiveArray::<i16>::new(), name),
            &Type::INT4 => Column::new(MutablePrimitiveArray::<i32>::new(), name),
            &Type::INT8 => Column::new(MutablePrimitiveArray::<i64>::new(), name),
            &Type::FLOAT4 => Column::new(MutablePrimitiveArray::<f32>::new(), name),
            &Type::FLOAT8 => Column::new(MutablePrimitiveArray::<f64>::new(), name),
            &Type::TIMESTAMP => Column::new(MutablePrimitiveArray::<i64>::new(), name),
            &Type::TIMESTAMPTZ => Column::new(MutablePrimitiveArray::<i64>::new(), name),
            &Type::DATE => Column::new(MutablePrimitiveArray::<i32>::new(), name),
            &Type::VARCHAR | &Type::CHAR_ARRAY | &Type::TEXT | &Type::NAME | &Type::UNKNOWN => {
                Column::new(MutableBinaryArray::<i32>::new(), name)
            }
            _ => todo!(),
        };
        table.push(col)
    }
    Ok(table)
}

#[inline(always)]
fn append_row(table: &mut Vec<Column>, row: &pg::Row) -> Result<()> {
    for (idx, column) in table.iter_mut().enumerate() {
        match column.dtype() {
            &DataType::Binary => {
                column.push::<_, MutableBinaryArray<i32>>(row.get::<_, Option<Vec<u8>>>(idx))?
            }
            &DataType::Boolean => column.push::<_, MutableBooleanArray>(row.try_get(idx).ok())?,
            &DataType::Int8 => {
                column.push::<_, MutablePrimitiveArray<i8>>(row.try_get(idx).ok())?
            }
            &DataType::Int16 => {
                column.push::<_, MutablePrimitiveArray<i16>>(row.try_get(idx).ok())?
            }
            &DataType::Int32 => {
                column.push::<_, MutablePrimitiveArray<i32>>(row.try_get(idx).ok())?
            }
            &DataType::Int64 => {
                column.push::<_, MutablePrimitiveArray<i64>>(row.try_get(idx).ok())?
            }
            &DataType::Float32 => {
                column.push::<_, MutablePrimitiveArray<f32>>(row.try_get(idx).ok())?
            }
            &DataType::Float64 => {
                column.push::<_, MutablePrimitiveArray<f64>>(row.try_get(idx).ok())?
            }

            // TODO: Need to determine TZ (if exists) before here, probably in `init_table`
            DataType::Timestamp(t, tz) => {
                column.push::<_, MutablePrimitiveArray<i64>>(row.try_get(idx).ok())?
            }
            DataType::Date32 => {
                column.push::<_, MutablePrimitiveArray<i32>>(row.try_get(idx).ok())?
            }
            _ => todo!(),
        }
    }

    Ok(())
}

fn column_names(row: &pg::Row) -> impl Iterator<Item = String> + '_ {
    row.columns().iter().map(|c| c.name().to_string())
}
