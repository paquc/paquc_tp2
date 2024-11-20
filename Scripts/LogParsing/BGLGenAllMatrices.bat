clear

python .\BGLPreprocessData.py 60min 1
python .\BGLPreprocessData.py 45min 1
python .\BGLPreprocessData.py 30min 1
python .\BGLPreprocessData.py 20min 1
python .\BGLPreprocessData.py 10min 1
python .\BGLPreprocessData.py 5min 1

python .\GenCorrMatrix.py ./BGL_Brain_results/BGL_5min_1_alarm_occurences_matrix_preprocessed.csv
python .\GenCorrMatrix.py ./BGL_Brain_results/BGL_10min_1_alarm_occurences_matrix_preprocessed.csv
python .\GenCorrMatrix.py ./BGL_Brain_results/BGL_20min_1_alarm_occurences_matrix_preprocessed.csv
python .\GenCorrMatrix.py ./BGL_Brain_results/BGL_30min_1_alarm_occurences_matrix_preprocessed.csv
python .\GenCorrMatrix.py ./BGL_Brain_results/BGL_45min_1_alarm_occurences_matrix_preprocessed.csv
python .\GenCorrMatrix.py ./BGL_Brain_results/BGL_60min_1_alarm_occurences_matrix_preprocessed.csv


@REM python .\BGLPreprocessData.py 15min 10 
@REM python .\BGLPreprocessData.py 20min 10 
@REM python .\BGLPreprocessData.py 25min 10 

@REM python .\BGLGenOccurencesMatricesFixWindow.py KERNDTLB ALL 1000 50 1 
@REM python .\BGLGenOccurencesMatricesFixWindow.py KERNDTLB ALL 2000 50 1 
@REM python .\BGLGenOccurencesMatricesFixWindow.py KERNDTLB ALL 3000 45 1 
@REM python .\BGLGenOccurencesMatricesFixWindow.py KERNDTLB ALL 4500 15 1 
@REM python .\BGLGenOccurencesMatricesFixWindow.py KERNDTLB ALL 5000 15 1 
@REM python .\BGLGenOccurencesMatricesFixWindow.py KERNDTLB ALL 5001 50 1 
