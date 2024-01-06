@echo off
set PYTHON=.\venv\Scripts\python.exe
set HF_HOME=huggingface
set TF_ENABLE_ONEDNN_OPTS=0

chcp 65001 > nul
set /p userInput=请输入图片文件夹路径: 
chcp 437 > nul

%PYTHON% tagger.py --image_folder=%userInput%

pause
