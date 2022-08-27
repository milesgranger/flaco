//#![warn(missing_docs)]
use postgres as pg;
use postgres::fallible_iterator::FallibleIterator;
use postgres::types::{FromSql, Type};
use postgres::RowIter;
use rust_decimal::prelude::{Decimal, ToPrimitive};
use std::any::Any;
use std::error::Error;
use std::iter::Iterator;
use std::net::IpAddr;
use std::collections::{HashMap};
use time;
use arrow2::{array::{Array, MutablePrimitiveArray, MutableArray}, array, datatypes::DataType};


type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub type Table = Vec<Column>;

pub struct Column {
    array: Box<dyn Array>,
    dtype: DataType,
    name: String
}
impl Column  {
    pub fn new(array: impl MutableArray, name: impl ToString) -> Self {
        let mut array = array;
        Self {
            dtype: array.data_type().clone(),
            array: array.as_box(),
            name: name.to_string()
        }
    }
    pub fn dtype(&self) -> &DataType {
        &self.dtype
    }
    pub fn inner_mut<T: Any + 'static>(&mut self) -> &mut T {
        self.array.as_any_mut().downcast_mut::<T>().unwrap()
    }
}

pub fn read_sql(client: &mut pg::Client, sql: &str) -> Result<Table> {
    let mut row_iter = client.query_raw::<_, &i32, _>(sql, &[])?;
    let mut table = if let Some(row) = row_iter.next()? {
        let mut table = init_table(&row)?;
        append_row(&mut table, &row)?;
        table
    } else {
        return Err("Query returned no rows!".into())
    };
    while let Some(row) = row_iter.next()? {
        append_row(&mut table, &row);
    }        
    todo!()
}

#[inline(always)]
fn init_table(row: &pg::Row) -> Result<Vec<Column>> {
    let mut table = Vec::with_capacity(row.len());
    for column in row.columns() {
        let col = match column.type_() {
            &pg::types::Type::BYTEA => {
                Column::new(array::MutableBinaryArray::<i32>::new(), column.name())
            },
            &pg::types::Type::CHAR => {
                Column::new(array::MutablePrimitiveArray::<i8>::new(), column.name())
            }
            _ => todo!()
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
                let arr = column.inner_mut::<array::MutableBinaryArray<i32>>();
                arr.push(row.get::<_, Option<Vec<u8>>>(idx));
            },
            &DataType::UInt8 => {
                let arr = column.inner_mut::<array::MutablePrimitiveArray<i8>>();
                arr.push(row.get::<_, Option<i8>>(idx));
            }
            _ => todo!()
        }
    }
    
    Ok(())
}

fn column_names(row: &pg::Row) -> impl Iterator<Item=String> + '_ {
    row.columns()
        .iter()
        .map(|c| c.name().to_string())
}
