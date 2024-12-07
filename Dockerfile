# 使用 Ubuntu 作為基底映像
FROM ubuntu:22.04

# 更新系統並安裝基礎工具
RUN apt-get update && apt-get install -y \
    wget \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 安裝 Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    /opt/conda/bin/conda init bash

# 添加 Conda 到 PATH
ENV PATH /opt/conda/bin:$PATH

# 設置清華 Conda 鏡像源
RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge && \
    conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/msys2 && \
    conda config --set show_channel_urls yes

# 複製 Conda yml 文件並建立環境
COPY aicup_cpu.yml /tmp/
RUN conda env create -f /tmp/aicup_cpu.yml && conda clean -a

# 安裝 Jupyter Notebook
RUN /opt/conda/envs/aicup/bin/pip install notebook && \
    mkdir /app/notebooks

# 設置執行時的 Conda 環境
ENV PATH /opt/conda/envs/aicup/bin:$PATH
ENV CONDA_DEFAULT_ENV aicup

# 暴露 Jupyter Notebook 的預設埠
EXPOSE 8888

# 設定工作目錄
WORKDIR /app

# 預設執行 Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
