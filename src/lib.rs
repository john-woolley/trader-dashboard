use pyo3::prelude::*;
use polars::prelude::*;
use std::collections::HashMap;
use tokio_postgres::{Client, NoTls, Row, Error};
use polars::datatypes::{DataType, Field};
use polars_arrow::array::{ArrayRef, Float32Array, Float64Array, Int32Array, Utf8Array, Int64Array};
use tokio_postgres::types::Type;
use tokio::runtime::Runtime;

enum RenderValue {
    Float(f64),
    Int(i64),
    String(String),
}


/// A Python module implemented in Rust.
#[pymodule]
fn celery_app(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Trader>()?;
    Ok(())
}
//
#[pyclass]
struct Trader {
    table_name: String,
    jobname: String,
    chunk: usize,
    initial_balance: f64,
    n_lags: usize,
    transaction_cost: f64,
    ep_length: usize,
    test: bool,
    risk_aversion: f64,
    render_mode: String,
    current_step: usize,
    render_dict: HashMap<String, HashMap<String, RenderValue>>,
    balance: f64,
    return_series: Vec<f64>,
    var: f64,
    corr: Vec<f64>,
    hursts: Vec<f64>,
    action: Vec<f64>,
    net_leverage: Vec<f64>,
    model_portfolio: Vec<f64>,
    spot: Vec<f64>,
    prev_spot: Vec<f64>,
    deviations: Vec<f64>,
    data: DataFrame,
}

async fn postgres_to_polars(rows: &[Row]) -> std::result::Result<DataFrame, PolarsError> {

    let mut fields: Vec<(String, polars::datatypes::DataType)> = Vec::new();
    let column_count = rows[0].len();
    for i in 0..column_count {
        let field_name = rows[0].columns()[i].name().to_string();
        let data_type: polars::datatypes::DataType = match rows[0].columns()[i].type_() {
            &Type::VARCHAR => polars::datatypes::DataType::String,
            &Type::TEXT => polars::datatypes::DataType::String,
            &Type::INT4 => polars::datatypes::DataType::Int32,
            &Type::INT8 => polars::datatypes::DataType::Int64,
            &Type::FLOAT4 => polars::datatypes::DataType::Float32,
            &Type::FLOAT8 => polars::datatypes::DataType::Float64,
            _ => panic!("Unsupported PostgreSQL data type"),
        };
        fields.push((field_name, data_type));
    }

    let first_row = rows.first().unwrap();

    let mut arrow_arrays: Vec<Vec<ArrayRef>> = vec![];
    
    for (col_index, column) in first_row.columns().iter().enumerate() {

        let mut array_data: Vec<ArrayRef> = vec![];

            for (row_index, row) in rows.iter().enumerate() {
        
                let array: Box<dyn polars_arrow::array::Array> = match column.type_() {
                    &Type::VARCHAR => Box::new(Utf8Array::<i64>::from(vec![Some(row.try_get::<usize, String>(col_index).expect("ErrVarChar"))])),
                    &Type::TEXT => Box::new(Utf8Array::<i64>::from(vec![Some(row.try_get::<usize, String>(col_index).expect("ErrString"))])),
                    &Type::INT4 => Box::new(Int32Array::from(vec![Some(row.try_get(col_index).expect("ErrInt32"))])),
                    &Type::INT8 => Box::new(Int64Array::from(vec![Some(row.try_get(col_index).expect("ErrInt64"))])),
                    &Type::FLOAT4 => Box::new(Float32Array::from(vec![Some(row.try_get(col_index).expect("ErrF32"))])),
                    &Type::FLOAT8 => Box::new(Float64Array::from(vec![Some(row.try_get(col_index).expect("ErrF64"))])),
                    _ => panic!("Unsupported PostgreSQL data type"),
                };
        
                array_data.push(array);
        
            }

            arrow_arrays.push(array_data);
    }

    let mut series: Vec<Series> = vec![];

    for (array, field) in arrow_arrays.iter().zip(fields.iter()) {
        // Debug print to inspect the Arrow array before Series creation
        println!("Arrow array for {}: {:?}", field.0, array);
    
        unsafe {
            let s = Series::from_chunks_and_dtype_unchecked(&field.0, array.to_vec(), &field.1)
                .fill_null(FillNullStrategy::Forward(None))?;
            
            // Debug print to inspect the Series data
            println!("Series for {}: {:?}", field.0, s);
            
            series.push(s);
        }
    }
    
    let df: PolarsResult<DataFrame> = DataFrame::new(series);
    

    df
}

async fn read_database(table_name: String) -> Result<DataFrame, Box<dyn std::error::Error>> {
    let (client, connection) = tokio_postgres::connect(
        "host=postgres user=trader_dashboard password=psltest dbname=trader_dashboard",
        NoTls,
    ).await?;
    tokio::spawn(async move {
        if let Err(e) = connection.await {
            eprintln!("connection error: {}", e);
        }
    });
    let rows: Vec<tokio_postgres::Row> = client.query(&format!("SELECT * FROM {}", table_name), &[]).await?;
    let df = postgres_to_polars(&rows).await?;
    let df_filled = df.select::<&[&str], _>(&[])?.fill_null(FillNullStrategy::Forward(None))?;
Ok(df_filled)
}

#[pymethods]
impl Trader {
    #[new]
    fn new(
        table_name: String,
        jobname: String,
        chunk: usize,
        initial_balance: f64,
        n_lags: usize,
        transaction_cost: f64,
        ep_length: usize,
        test: bool,
        risk_aversion: f64,
        render_mode: String,
    ) -> Self {
        let runtime = Runtime::new().unwrap();
        let data = runtime.block_on(read_database(table_name.clone())).expect("Error reading database");
        let render_dict = HashMap::new();
        let balance = initial_balance;
        let return_series = vec![];
        let var = 0.0;
        let corr = vec![];
        let hursts = vec![];
        let action = vec![];
        let net_leverage = vec![];
        let model_portfolio = vec![];
        let spot = vec![];
        let prev_spot = vec![];
        let deviations = vec![];
        let trader = Trader {
            table_name,
            jobname,
            chunk,
            initial_balance,
            n_lags,
            transaction_cost,
            ep_length,
            test,
            risk_aversion,
            render_mode,
            current_step: 0,
            render_dict,
            balance,
            return_series,
            var,
            corr,
            hursts,
            action,
            net_leverage,
            model_portfolio,
            spot,
            prev_spot,
            deviations,
            data
        };
        trader
    }

    fn reset(&mut self) {
        self.current_step = 0;
        self.balance = self.initial_balance;
        self.return_series = vec![];
        self.var = 0.0;
        self.corr = vec![];
        self.hursts = vec![];
        self.action = vec![];
        self.net_leverage = vec![];
        self.model_portfolio = vec![];
        self.spot = vec![];
        self.prev_spot = vec![];
        self.deviations = vec![];
    }
}