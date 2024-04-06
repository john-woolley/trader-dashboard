use pyo3::{ffi::printfunc, prelude::*, types::PyTuple};
use polars::prelude::*;
use std::collections::{HashMap, VecDeque};
use pyo3_polars::{PyDataFrame, PySeries};
use pyo3::types::IntoPyDict;
use chrono::NaiveDate;
use polars::prelude::*;

enum RenderValue {
    Float(f64),
    Int(i64),
    String(String),
}

impl IntoPy<PyObject> for RenderValue {
    fn into_py(self, py: Python) -> PyObject {
        match self {
            RenderValue::Float(f) => f.into_py(py),
            RenderValue::Int(i) => i.into_py(py),
            RenderValue::String(s) => s.into_py(py),
        }
    }
}

enum RenderMode {
    Live,
    File,
}

enum StepResult {
    Observation(Vec<f64>),
    Reward(f64),
    Done(bool),
    Bool(bool),
    Info(HashMap<String, RenderValue>),
}

const USED_COLS: [&str; 13] = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ebitdamargin",
    "fcfps",
    "pb",
    "ps",
    "pe",
    "de",
    "capex",
    "revenue",
];

const macro_cols: [&str; 39] = [
    "CPILFESL",
    "CPIENGSL",
    "TTLCONS",
    "JTSJOR",
    "JTSLDR",
    "JTSQUR",
    "JTSHIR",
    "GDP",
    "UNRATE",
    "CIVPART",
    "CES0500000003",
    "AWHAETP",
    "PAYEMS",
    "ISRATIO",
    "DGS10",
    "M2REAL",
    "BAA10Y",
    "DFF",
    "DEXJPUS",
    "DEXUSEU",
    "EURJPY",
    "DEXUSUK",
    "GBPJPY",
    "DEXSZUS",
    "CHFJPY",
    "INTGSTJPM193N",
    "IRSTCI01CHM156N",
    "IUDSOIA",
    "IR3TIB01DEM156N",
    "IRLTLT01DEM156N",
    "IRLTLT01FRM156N",
    "IRLTLT01ITM156N",
    "ECBASSETSW",
    "T10Y2Y",
    "JPNASSETS",
    "IRLTLT01JPM156N",
    "IRLTLT01GBM156N",
    "WTISPLC",
    "DEXCAUS",
];



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
    paid_slippage: f64,
    rsi: Vec<f64>,
    state_buffer: VecDeque<Vec<f64>>,
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
        let paid_slippage = 0.0;
        let rsi = vec![];
        let state_buffer = VecDeque::new();

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
            dates,
            paid_slippage,
            rsi,
            state_buffer,
        };
        trader
    }

    fn reset(&mut self) -> (Vec<f64>, HashMap::<String, RenderValue>){
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
        println!("Date: {}", predicate.clone());
        let predicate_date = NaiveDate::parse_from_str(&predicate, "%Y-%m-%d").unwrap();
        let filter_expr = col("date").eq(lit(predicate_date));
        self.spot = self.data
            .select(&["spot", "date"])
            .unwrap()
            .lazy()
            .filter(filter_expr)
            .collect()
            .unwrap()
            .select(&["spot"])
            .unwrap()
            .column("spot")
            .unwrap()
            .f64()
            .unwrap()
            .into_no_null_iter()
            .collect();
        println!("Spot: {:?}", self.spot);
        self.prev_spot = self.spot.clone();
        self.deviations = vec![];
        self.paid_slippage = 0.0;
        self.rsi = vec![50.0; self.no_symbols];
        self.state_buffer = VecDeque::new();
        let state_frame = self._get_state_frame();
        let state_frame_rs = Into::<DataFrame>::into(state_frame);
        let mut vec = Vec::new();
        for name in state_frame_rs.get_column_names() {
            let series = state_frame_rs.column(name).unwrap();
            if let Ok(floats) = series.f64() {
                vec.extend(floats.into_iter().filter_map(|x| x));
            }
        }
        while self.state_buffer.len() < self.n_lags {
            self.state_buffer.push_back(vec.clone());
        }
        return (self._get_observation(), HashMap::new());
    }   


    fn _get_state_frame(&self) -> PyDataFrame {
        let predicate_date = NaiveDate::parse_from_str(&self.dates.get(self.current_step).unwrap().to_string(), "%Y-%m-%d").unwrap();
        let mut state_frame = self.data
            .select(&["spot", "date", "ticker"])
            .unwrap()
            .lazy()
            .filter(col("date").eq(lit(predicate_date)))
            .collect()
            .unwrap();

        if self.current_step < self.n_lags {
            for _ in 1..self.current_step {
                let predicate: String = self.dates.get(self.current_step).unwrap().to_string();
                let filter_expr = col("date").eq(lit(predicate));
                let lag_frame = self.data
                    .select(&["spot", "date", "ticker"])
                    .unwrap()
                    .lazy()
                    .filter(filter_expr)
                    .collect()
                    .unwrap();
                state_frame = state_frame.vstack(&lag_frame).unwrap();
            }
            return PyDataFrame(state_frame);
        }
        for i in 1..self.n_lags {

            let predicate: String = self.dates.get(self.current_step - i).unwrap().to_string();
            let filter_expr = col("date").eq(lit(predicate));
            let lag_frame = self.data
                .select(&["spot", "date", "ticker"])
                .unwrap()
                .lazy()
                .filter(filter_expr)
                .collect()
                .unwrap();
            state_frame = state_frame.vstack(&lag_frame).unwrap();

        }
        let res = PyDataFrame(state_frame);
        res
    }

    fn _get_hurst_exponent(&self, series: Vec<f64>) -> f64 {
        let n = series.len();
        let mut tau = vec![];
        let mut lagvec = vec![];
        let mut lags = (n / 2).min(300);
        let mut i = 0;
        while lags > 2 {
            let mut cc = 0.0;
            let mut mean = 0.0;
            for j in 0..n - lags {
                mean += series[j];
                cc += (series[j + lags] - series[j]).powi(2);
            }
            cc = cc / (n - lags) as f64;
            mean = mean / (n - lags) as f64;
            tau.push((cc / mean).sqrt());
            lagvec.push(lags as f64);
            lags = (lags as f64 * 0.5) as usize;
            i += 1;
        }
        let slope = (tau[0].log10() - tau[tau.len() - 1].log10()) / (lagvec[0].log10() - lagvec[lagvec.len() - 1].log10());
        slope
    }

    fn _get_rsi(&self, series: Vec<f64>) -> f64 {
        let mut gain = 0.0;
        let mut loss = 0.0;
        for i in 1..series.len() {
            let diff = series[i] - series[i - 1];
            if diff > 0.0 {
                gain += diff;
            } else {
                loss -= diff;
            }
        }
        let rs = gain / loss;
        100.0 - (100.0 / (1.0 + rs))
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

    #[getter]
    fn model_portfolio_value(&self) -> f64 {
        self.model_portfolio.clone().iter().sum()
    }

    fn get_net_position_weights(&self) -> Vec<f64> {
        let net_position_values = self.get_net_position_values();
        let total_net_position_value: f64 = net_position_values.iter().sum();
        net_position_values.iter().map(|x| x / total_net_position_value).collect()
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

    fn _trade(&mut self, amount: f64, sign: f64, underlying: usize) -> () {
        let paid_slippage = self.transaction_cost * amount * self.spot[underlying];
        self.net_leverage[underlying] = self.net_leverage[underlying] + sign * amount;
        self.balance = self.balance - sign * amount * self.spot[underlying] - paid_slippage;
        self.paid_slippage = paid_slippage;
    }

    fn _get_observation(&self) -> Vec<f64> {
        let obs = self.state_buffer.iter().flatten().copied().collect::<Vec<f64>>();
        obs
    }

    fn _accrue_interest(&mut self) -> () {
        let interest = self.get_total_net_position_value() * 0.01 / 252.0;
        self.balance = self.balance + interest;
    }

    fn _yeo_johnson_transform(&self, val: f64, lambda: f64) -> f64{
        if lambda == 0.0 {
            val.ln()
        } else {
            if val < 0.0 {
                -((-val + 1.0).powf(lambda) - 1.0) / lambda
            } else {
                (val + 1.0).powf(lambda) - 1.0
            }
        }
    }

    fn _get_return(&self, current_portfolio_value: f64, prev_portfolio_value: f64) -> f64 {
        (current_portfolio_value - prev_portfolio_value) / prev_portfolio_value
    }

    fn _get_reward(&self, current_portfolio_value: f64, prev_portfolio_value: f64) -> f64 {
        let ret = self._get_return(current_portfolio_value, prev_portfolio_value);
        let rf_ret = 0.01 / 252.0;
        let ret_vol = self.return_series.iter().fold(0.0, |acc, x| acc + x.powi(2)).sqrt();
        let reward = self._yeo_johnson_transform((ret-rf_ret) / ret_vol, self.risk_aversion);
        let normalized_reward = 1.0 / (1.0 + (-reward).exp());
        normalized_reward
    }

    fn step(&mut self, action: Vec<f64>) -> (Vec<f64>, f64, bool, bool, HashMap<String, RenderValue>){
        self.paid_slippage = 0.0;
        let prev_portfolio_value = self.get_current_portfolio_value();
        self.prev_spot = self.spot.clone();
        self._accrue_interest();

        for i in 0..self.no_symbols {
            self.model_portfolio[i] = action[i];
        }

        let short_leverage = action[action.len() - 3];
        let long_leverage = action[action.len() - 2];
        let trade_today = action[action.len() - 1] > 0.0 || self.current_step % 20 == 0;
        let target_weights = self.get_model_portfolio_weights();

        let mut deviations: Vec<f64> = target_weights.iter()
                    .zip(&self.model_portfolio)
                    .map(|(target, model)| target - model)
                    .collect();

        let net_position_weights = self.get_net_position_weights();

        for i in 0..self.no_symbols {
            let actual_weight = net_position_weights[i];
            let target_weight = target_weights[i];
            deviations[i] = target_weight - actual_weight;
        let sum_deviation: f64 = deviations.iter().sum();
        deviations = deviations.iter().map(|x| x / sum_deviation).collect();
        
        if trade_today {
            for i in 0..self.no_symbols {
                let deviation = deviations[i];
                let sign = deviation.signum();
                let amount = deviation.abs();
                if sign > 0.0 {
                    self._trade(amount, sign, i);
                }
            }
        }
    }

        let obs = self._get_observation();
        let current_portfolio_value = self.get_current_portfolio_value();
        let reward = self._get_reward(current_portfolio_value, prev_portfolio_value);
        self.current_step += 1;
        let done = self.current_step >= self.ep_length || current_portfolio_value < 0.5 * self.initial_balance;

        let res = (
            obs,
            reward,
            done,
            false,
            HashMap::new(),
        );
        return res;
    }
}