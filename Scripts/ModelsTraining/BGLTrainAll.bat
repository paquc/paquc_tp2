clear
@REM python .\BGL_Train_VAPI_Alarms_FixWindow.py 1 1 ALL 1000 0 1 80 20 0 KERNDTLB
@REM python .\BGL_Train_VAPI_Alarms_FixWindow.py 1 1 ALL 2000 0 1 80 20 0 KERNDTLB
@REM python .\BGL_Train_VAPI_Alarms_FixWindow.py 1 1 ALL 3000 0 1 80 20 0 KERNDTLB
@REM python .\BGL_Train_VAPI_Alarms_FixWindow.py 1 1 ALL 4500 0 1 80 20 0 KERNDTLB
@REM python .\BGL_Train_VAPI_Alarms_FixWindow.py 1 1 ALL 5000 0 1 80 20 0 KERNDTLB
@REM python .\BGL_Train_VAPI_Alarms_FixWindow.py 1 1 ALL 5001 0 1 80 20 0 KERNDTLB

python .\BGL_Train_Alarms_Processed.py 1 1 60min 1 0 1 80 20 0 corr
python .\BGL_Train_Alarms_Processed.py 1 1 30min 1 0 1 80 20 0 corr
python .\BGL_Train_Alarms_Processed.py 1 1 45min 1 0 1 80 20 0 corr
python .\BGL_Train_Alarms_Processed.py 1 1 20min 1 0 1 80 20 0 corr
python .\BGL_Train_Alarms_Processed.py 1 1 10min 1 0 1 80 20 0 corr
python .\BGL_Train_Alarms_Processed.py 1 1 5min 1 0 1 80 20 0 corr



