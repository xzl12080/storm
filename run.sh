#!/bin/bash

proxy_on() {
    export http_proxy=http://127.0.0.1:7890
    export https_proxy=http://127.0.0.1:7890
    export no_proxy=127.0.0.1,localhost
    export HTTP_PROXY=http://127.0.0.1:7890
    export HTTPS_PROXY=http://127.0.0.1:7890
    export NO_PROXY=127.0.0.1,localhost
    echo -e "\033[32m[√] 已开启代理\033[0m"
}

proxy_on

export http_proxy
export https_proxy
export no_proxy
export HTTP_PROXY
export HTTPS_PROXY
export NO_PROXY

# # 启动 Python 服务
# nohup python fake_api.py --server-name storm --port 26300 > fake_api.log 2>&1 &
nohup python api.py --server-name storm --port 26300 > api.log 2>&1 &
# python api_v1_1.py --server-name storm --port 26302