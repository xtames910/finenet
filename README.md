# FineNet on ACB Stock — Forecasting Anomalous Price Events

This is a small reimplementation of **FineNet** (Tsai et al., RecSys '19), a CNN+GRU model that
tries to catch abnormal stock movements before they happen. Instead of the 300 companies used in
the original paper, I ran it on a single stock — **ACB (Asia Commercial Bank)** — using its daily
price history from 2017 onward.

The core idea, taken straight from the paper: for a given week, compute the ROI compared to its
historical mean/std. If it's more than `k` standard deviations away (either up or down), that week
counts as an "anomalous event." The model's job is to predict next week's event ahead of time, so
you'd get a heads-up before the stock does something extreme.

## What FineNet actually does

It's a multi-branch CNN feeding into a BiGRU:

- Several Conv1D branches run in parallel over the price window — some with dilation, so the model
  can pick up both short-term and longer-term patterns without stacking a ton of layers.
- The branches get concatenated and passed through a Bidirectional GRU to learn the sequential part.
- Two output heads at the end: one regresses the next value, the other classifies whether it's an
  anomaly. Multi-task, trained together.

## Data

`ACB.csv` has daily OHLCV data plus some pre-computed columns — weekly ROI, rolling mean/std, and
flags for whether each row is an anomaly at k=0.5 / 1.0 / 1.5. The notebook uses closing price and
weekly ROI, scales them with MinMaxScaler, and builds sliding windows of size 10 as input. Split is
80/20, kept in time order (no shuffling — it's a time series, so shuffling would leak the future).

## Baselines

Same comparison as the paper: ARIMA, a plain GRU, and a stacked-LSTM "DRNN," all predicting the same
target and thresholded the same way (k·σ from the mean).

## Results

Ran everything on ACB's test split, three k values:

| Model | Prec (k=0.5) | Prec (k=1.0) | Prec (k=1.5) | Rec (k=0.5) | Rec (k=1.0) | Rec (k=1.5) |
|---|---|---|---|---|---|---|
| ARIMA | 0.49 | 0.45 | 0.38 | 0.64 | 0.25 | 0.08 |
| GRU | 0.55 | 0.56 | 0.50 | 0.65 | 0.32 | 0.12 |
| DRNN | 0.48 | 0.56 | 0.59 | 0.61 | 0.31 | 0.12 |
| **FineNet** | 0.54 | 0.50 | **0.67** | **0.90** | **0.94** | **0.56** |

The gap is mostly in recall — FineNet catches way more of the actual anomalies than the other three,
sometimes 2-3x more, without giving up much on precision. That lines up with what the paper claims.
Makes sense the numbers aren't identical to the paper's (single stock vs. 300 companies is a very
different setup), but the pattern — FineNet > GRU/DRNN > ARIMA, especially as k grows — holds up.

## Heads up on a couple of rough edges

- The "voting system" cell near the end of the notebook (combining predictions from all four models)
  is currently running on **randomly generated fake data**, not the actual model outputs from
  earlier in the notebook. It's there to show how the voting logic works, but the numbers it prints
  don't mean anything yet — would need to be wired up to the real predictions.
- The original ARIMA loop feeds its own forecasts back into the input history, which can blow up and
  fail to converge after enough steps. I reworked it to do a proper walk-forward using real past
  prices instead — that's what the numbers above reflect.
- Notebook was written for Colab (`files.upload()`), so swap that line for `pd.read_csv('ACB.csv')`
  if running locally.

## Stack

pandas, numpy, scikit-learn, statsmodels (ARIMA), TensorFlow/Keras (Conv1D, GRU, LSTM, Bidirectional)

## Running it

```bash
pip install pandas numpy matplotlib seaborn scikit-learn statsmodels tensorflow
jupyter notebook FINENET.ipynb
```

## Reference

Tsai, Y.-C., Chen, C.-Y., Ma, S.-L., Wang, P.-C., Chen, Y.-J., Chang, Y.-C., & Li, C.-T. (2019).
*FineNet: A Joint Convolutional and Recurrent Neural Network Model to Forecast and Recommend
Anomalous Financial Items.* RecSys '19. https://doi.org/10.1145/3298689.3346968
