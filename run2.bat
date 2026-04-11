@echo off
cd /d E:\Garry\market_model
call .venv\Scripts\activate

python "src\python prototype_formula_ablation.py" --csv data\ta35_model_data.csv --output-dir output\ablation_default
python "src\python prototype_formula_ablation.py" --csv data\ta35_model_data.csv --threshold-mode quantile --threshold-quantile 0.7 --output-dir output\ablation_quantile_07

pause
