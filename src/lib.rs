//#![warn(missing_docs)]
use postgres as pg;
use postgres::fallible_iterator::FallibleIterator;
use postgres::types::{FromSql, Type};
use postgres::RowIter;
use rust_decimal::prelude::{Decimal, ToPrimitive};
use std::error::Error;
use std::iter::Iterator;
use std::net::IpAddr;
use std::collections::{HashMap};
use time;
use arrow2::{array::Array, array, datatypes::DataType};


type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;
pub type Table = Vec<Column>;

pub struct Column {
    array: Box<dyn Array>,
    dtype: DataType,
    name: String
}
impl Column  {
    pub fn new(array: impl Array, name: impl ToString) -> Self {
        Self {
            dtype: array.data_type().clone(),
            array: array.to_boxed(),
            name: name.to_string()
        }
    }
    pub fn dtype(&self) -> &DataType {
        &self.dtype
    }
    pub fn append<T>(&mut self, val: T) -> Result<()> {
        todo!()
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
                Column::new(array::BinaryArray::<i32>::new_empty(DataType::Binary), column.name())
            },
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
            &DataType::Binary => column.append(row.get::<_, Option<Vec<u8>>>(idx))?,
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
