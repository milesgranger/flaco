//#![warn(missing_docs)]

use tokio_postgres as pg;

/// Supports creating connections to a given connection URI
pub struct Engine {
    client: pg::Client,
}

impl Engine {
    /// Create an `Engine` object
    /// `uri` is to conform to any of the normal connection strings, described
    /// in more [detail here](https://docs.rs/tokio-postgres/0.7.2/tokio_postgres/config/struct.Config.html#examples)
    pub async fn new(uri: &str) -> Self {
        let (client, con) = pg::connect(uri, pg::NoTls).await.unwrap();
        tokio::spawn(async move {
            if let Err(e) = con.await {
                eprintln!("connection error: {}", e);
            }
        });
        Self { client }
    }
}


#[repr(C)]
pub enum Data {
    Int64(i64),
}


#[no_mangle]
pub extern "C" fn read_sql() -> Data {
    // read query to start rowstream

    // get first row, and construct schema/columns in numpy

    // iterate over each row in the stream

        // for each column value in row

            // First iteration, check if arrays should be resized to fit new row.

            // if value is None, convert to appropriate pandas null type (pd.NA, pd.NaT)

            // insert element into array
    Data::Int64(1)
}


#[cfg(test)]
mod tests {

    use super::*;
    const CONNECTION_URI: &str = "postgresql://postgres:postgres@localhost:5432/postgres";

    #[tokio::test]
    async fn basic_query() {
        let engine = Engine::new(CONNECTION_URI).await;
        engine
            .execute("create table if not exists foobar (col1 integer, col2 integer)")
            .await;
        let n_rows = engine
            .execute("insert into foobar (col1, col2) values (1, 1)")
            .await;
        assert_eq!(n_rows, 1)
    }
}
