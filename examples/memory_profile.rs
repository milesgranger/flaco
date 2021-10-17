use flaco::*;
use std::ptr;
use std::ffi::CString;


fn main() {

    let mut db = Database::new("postgresql://postgres:postgres@localhost:5432/postgres");
    db.connect().unwrap();
    let db_ptr = Box::into_raw(Box::new(db));
    let mut row_iter = ptr::null_mut();
    let mut column_names = ptr::null_mut();
    let mut n_columns = 0;
    let mut row_data_array_ptr = ptr::null_mut();
    let mut exc = ptr::null_mut();
    let stmt = CString::new("select * from test_table").unwrap().into_raw();

    row_iter = read_sql(stmt, db_ptr as _, &mut exc);
    let mut n_rows = 0;
    loop {
        let _: () = next_row(&mut row_iter, &mut row_data_array_ptr, &mut n_columns, &mut column_names, &mut exc);
        if row_iter.is_null() {
            break
        }
        n_rows += 1;
        for i in 0..n_columns {
            let _data_ptr: *mut Data = index_row(row_data_array_ptr, n_columns, i);
        }
    }

    println!("Read {} rows!", n_rows);

}