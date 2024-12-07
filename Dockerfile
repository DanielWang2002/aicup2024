# 基於官方 Python 3.11 slim 基底映像
FROM python:3.11.10-slim

# 設定工作目錄
WORKDIR /app

# 安裝基礎套件及 Python 工具
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# 複製 requirements.txt 並安裝依賴
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 同步本地目錄，設置為可讀寫
VOLUME ["/app"]

# 預設指令，啟動 bash 供互動使用
CMD ["bash"]
