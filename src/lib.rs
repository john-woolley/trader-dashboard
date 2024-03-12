use pyo3::prelude::*;
use polars::prelude::*;
use std::collections::HashMap;
use pyo3_polars::{PyDataFrame, PySeries};

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
    data: DataFrame,
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
    no_symbols: usize,
    dates: Series,
}

#[pymethods]
impl Trader {
    #[new]
    fn new(
        pydata: PyDataFrame,
        initial_balance: f64,
        n_lags: usize,
        transaction_cost: f64,
        ep_length: usize,
        test: bool,
        risk_aversion: f64,
        render_mode: String,
    ) -> Self {
        let data: DataFrame = <PyDataFrame as Into<DataFrame>>::into(pydata);
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
        let no_symbols = 
            data.column("ticker")
                .unwrap()
                .unique()
                .unwrap()
                .len();
        let dates = 
            data.column("date")
                .unwrap()
                .unique()
                .unwrap();

        let trader = Trader {
            data,
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
            no_symbols,
            dates
        };
        trader
    }

    fn reset(&mut self) {
        self.current_step = 0;
        self.balance = self.initial_balance;
        self.return_series = vec![];
        self.var = 0.0;
        self.corr = vec![1.0; self.no_symbols];
        self.hursts = vec![0.5; self.no_symbols];
        self.action = vec![];
        self.net_leverage = vec![0.0; self.no_symbols];
        self.model_portfolio = vec![0.0; self.no_symbols];
        let predicate: String = self.dates.get(self.current_step).unwrap().to_string();
        let filter_expr = col("date").eq(lit(predicate));
        self.spot = self.data
            .select(&["spot", "date"])
            .unwrap()
            .lazy()
            .filter(filter_expr)
            .collect()
            .unwrap()
            .drop_in_place("date")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        self.prev_spot = self.spot.clone();
        self.deviations = vec![];
    }

    #[getter]
    fn get_no_symbols(&self) -> usize {
        self.no_symbols
    }

    #[getter]
    fn get_dates(&self) -> PySeries {
        PySeries(self.dates.clone())
    }

    #[getter]
    fn get_current_step(&self) -> usize {
        self.current_step
    }

    #[getter]
    fn get_balance(&self) -> f64 {
        self.balance
    }

    #[getter]
    fn get_total_net_position_value(&self) -> f64 {
        self.get_net_position_values().iter().sum()
    }

    #[getter]
    fn get_net_position_values(&self) -> Vec<f64> {
        self.net_leverage
        .iter()
        .zip(&self.spot)
        .map(|(leverage, spot)| leverage * spot)
        .collect()
    }

    #[getter]
    fn get_current_portfolio_value(&self) -> f64 {
        self.balance + self.get_total_net_position_value()
    }

    #[getter]
    fn get_total_gross_position_value(&self) -> f64 {
        self.get_gross_position_values().iter().sum()
    }

    #[getter]
    fn get_gross_position_values(&self) -> Vec<f64> {
        self.net_leverage
        .iter()
        .zip(&self.spot)
        .map(|(leverage, spot)| (leverage.abs()) * spot)
        .collect()
    }

    #[getter]
    fn get_current_date(&self) -> String {
        self.dates.get(self.current_step).unwrap().to_string()
    }

    fn get_model_portfolio_weights(&self) -> Vec<f64> {
        let scaled = self.model_portfolio.clone();

        let shorts: Vec<f64> = scaled.iter().copied().filter(|&x| x < 0.0).collect();
        let longs: Vec<f64> = scaled.iter().copied().filter(|&x| x >= 0.0).collect();

        let short_weights: Vec<f64> = shorts.iter().map(|x| x.abs()).collect();
        let long_weights: Vec<f64> = longs.iter().map(|x| x.abs()).collect();

        let short_sum: f64 = short_weights.iter().sum();
        let long_sum: f64 = long_weights.iter().sum();

        let res = scaled.iter().map(|&x| {
            if x > 0.0 {
                x / long_sum
            } else {
                x / short_sum
            }
        }).collect();
        res
    }



}