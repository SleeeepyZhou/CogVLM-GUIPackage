$Env:HF_HOME = "huggingface"
$Env:PIP_DISABLE_PIP_VERSION_CHECK = 1
$Env:PIP_NO_CACHE_DIR = 1
$Env:PIP_INDEX_URL = "https://mirror.baidu.com/pypi/simple"
$Env:HF_ENDPOINT = "https://hf-mirror.com"
function InstallFail {
    Write-Output "安装失败。"
    Read-Host | Out-Null ;
    Exit
}

function Check {
    param (
        $ErrorInfo
    )
    if (!($?)) {
        Write-Output $ErrorInfo
        InstallFail
    }
}

if (!(Test-Path -Path "venv")) {
    Write-Output "正在创建虚拟环境..."
    python -m venv venv
    Check "创建虚拟环境失败，请检查 python 是否安装完毕以及 python 版本是否为64位版本的python 3.10、或python的目录是否在环境变量PATH内。"
}

.\venv\Scripts\activate
Check "激活虚拟环境失败。"

$install_torch = Read-Host "是否需要安装 Torch? [y/n] (默认为 y)"
if ($install_torch -eq "y" -or $install_torch -eq "Y" -or $install_torch -eq ""){
    pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html -i https://mirror.baidu.com/pypi/simple
    Check "torch 安装失败，请删除 venv 文件夹后重新运行。"
	pip install -U -I --no-deps xformers==0.0.22
    Check "xformers 安装失败。"
}

Write-Output "安装 bitsandbytes..."
pip install bitsandbytes==0.41.1 --index-url https://jihulab.com/api/v4/projects/140618/packages/pypi/simple
Write-Output "安装 deepspeed..."
pip install deepspeed-0.11.2+8ce7471-py3-none-any.whl

Write-Output "安装依赖..."
pip install huggingface_hub
pip install scipy networkx wordcloud matplotlib Pillow tqdm gradio requests
pip install -r require.txt
Check "依赖安装失败。"

$download_get = Read-Host "是否需要下载模型(已配置国内加速)?模型大小约32G，如果有条件可以手动下载 [y/n] (默认为 y)"
if ($download_get -eq "y" -or $download_get -eq "Y" -or $download_get -eq ""){
    python download.py
}

Write-Output "安装完毕"
Read-Host | Out-Null ;
