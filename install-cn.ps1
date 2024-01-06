$Env:HF_HOME = "huggingface"
$Env:PIP_DISABLE_PIP_VERSION_CHECK = 1
$Env:PIP_NO_CACHE_DIR = 1
$Env:PIP_INDEX_URL = "https://mirror.baidu.com/pypi/simple"
$Env:HF_ENDPOINT = "https://hf-mirror.com"
function InstallFail {
    Write-Output "��װʧ�ܡ�"
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
    Write-Output "���ڴ������⻷��..."
    python -m venv venv
    Check "�������⻷��ʧ�ܣ����� python �Ƿ�װ����Լ� python �汾�Ƿ�Ϊ64λ�汾��python 3.10����python��Ŀ¼�Ƿ��ڻ�������PATH�ڡ�"
}

.\venv\Scripts\activate
Check "�������⻷��ʧ�ܡ�"

$install_torch = Read-Host "�Ƿ���Ҫ��װ Torch? [y/n] (Ĭ��Ϊ y)"
if ($install_torch -eq "y" -or $install_torch -eq "Y" -or $install_torch -eq ""){
    pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 -f https://mirror.sjtu.edu.cn/pytorch-wheels/torch_stable.html -i https://mirror.baidu.com/pypi/simple
    Check "torch ��װʧ�ܣ���ɾ�� venv �ļ��к��������С�"
	pip install -U -I --no-deps xformers==0.0.22
    Check "xformers ��װʧ�ܡ�"
}

Write-Output "��װ bitsandbytes..."
pip install bitsandbytes==0.41.1 --index-url https://jihulab.com/api/v4/projects/140618/packages/pypi/simple
Write-Output "��װ deepspeed..."
pip install deepspeed-0.11.2+8ce7471-py3-none-any.whl

Write-Output "��װ����..."
pip install huggingface_hub
pip install scipy networkx wordcloud matplotlib Pillow tqdm gradio requests
pip install -r require.txt
Check "������װʧ�ܡ�"

$download_get = Read-Host "�Ƿ���Ҫ����ģ��(�����ù��ڼ���)?ģ�ʹ�СԼ32G����������������ֶ����� [y/n] (Ĭ��Ϊ y)"
if ($download_get -eq "y" -or $download_get -eq "Y" -or $download_get -eq ""){
    python download.py
}

Write-Output "��װ���"
Read-Host | Out-Null ;
