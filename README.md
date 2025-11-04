# Time-Series-Analysis-for-Bitcoin-Price-Prediction-using-RNN-LSTM-and-GRU.
Bitcoin Price Prediction using RNN, LSTM, and GRU
**Project Overview**

This project performs time series forecasting of Bitcoin close prices using sequential deep learning models: SimpleRNN, LSTM, and GRU. The analysis uses the uploaded CSV file (btcusd_1-min_data.csv) which contains minute-level historical Bitcoin data; the pipeline resamples to daily close prices and trains models using a sliding-window approach (past 60 days → next day prediction).

**Requirements**

Install the required Python packages before running the script:

pip install pandas numpy matplotlib scikit-learn tensorflow

Note: TensorFlow is required to train the models. If you don't have GPU, CPU training is fine but will be slower.

**Quick Summary of Workflow**

Data loading & parsing — parse timestamp column and identify price column (Close).
Resample to daily — aggregate minute-level rows to daily using the last recorded price per day as Close.
EDA — time series plot, daily returns, moving averages.
Preprocessing — MinMax scale Close price; generate sliding-window sequences (window=60 days).
Train/Validation split — 80% train, 20% validation (time-ordered).
Models — train SimpleRNN, LSTM, and GRU (small architectures included).
Evaluation — compute RMSE & MAE, plot training loss and predicted vs actual.
Save outputs — models, plots, and summary JSON.

**Project Limitations & Next Steps**

Minute-level data is large; we resample to daily to reduce compute. For higher-frequency forecasting, adapt the sequence length and model batch sizes.
Consider adding features (Open, High, Low, Volume, technical indicators) to improve forecasts.
Use walk-forward cross-validation for robust time-series evaluation; implement hyperparameter tuning (Optuna, KerasTuner).
Be aware: financial time series are noisy and often non-stationary — forecasts should be used cautiously.



