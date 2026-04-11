@echo off
cd /d E:\Garry\market_model
call .venv\Scripts\activate

python src\prototype_formula_model_1.py --csv data\ta35_model_data.csv --output output\prototype_formula_backtest_default.csv
python src\prototype_formula_model_1.py --csv data\ta35_model_data.csv --threshold-mode quantile --threshold-quantile 0.7 --output output\prototype_formula_backtest_quantile_07.csv
python src\prototype_formula_model_1.py --csv data\ta35_model_data.csv --w-gap 1.0 --w-dual-ret 0.5 --w-frac-up 0.2 --output output\prototype_formula_backtest_weights_gap1_dual05_frac02.csv

pause
