FROM ubuntu:22.04

# 安装基础工具包
RUN apt-get update && \
    apt-get install -y git rsync jq git-lfs vim curl wget unzip lsof nload htop net-tools dnsutils openssh-server && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# 安装Miniconda
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh   -o miniconda.sh && \
    /bin/bash miniconda.sh -b -p /opt/miniconda3 && \
    rm miniconda.sh && \
    /opt/miniconda3/bin/conda clean --all

# 设置基础环境变量
ENV PATH="/opt/miniconda3/bin:$PATH"

# 添加频道并接受服务条款
RUN conda config --add channels https://repo.anaconda.com/pkgs/main   && \
    conda config --add channels https://repo.anaconda.com/pkgs/r   && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main   && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r  

# 创建Python 3.9环境并设置为默认
RUN conda create -n py39 python=3.9 -y && \
    conda clean --all


# 更新PATH优先使用Python 3.9环境
ENV PATH="/opt/miniconda3/envs/py39/bin:${PATH}"

# 创建符号链接
RUN ln -sf /opt/miniconda3/envs/py39/bin/python /usr/local/bin/python && \
    ln -sf /opt/miniconda3/envs/py39/bin/pip /usr/local/bin/pip

# 卸载numpy 2.0.1并安装指定版本的包
RUN ["/bin/bash", "-c", "source /opt/miniconda3/etc/profile.d/conda.sh && \
    conda activate py39 && \
    conda install -y numpy=1.23.5 pandas=1.3.4 matplotlib=3.5.0 scikit-learn=0.24.2 && \
    pip install torchsummary==1.5.1"]
    
# 在py39环境中安装PyTorch
RUN /bin/bash -c "source /opt/miniconda3/etc/profile.d/conda.sh && \
    conda activate py39 && \
    conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -y && \
    conda clean --all"

# 替换为以下代码
RUN echo "source /opt/miniconda3/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate py39" >> ~/.bashrc && \
    /bin/bash -c "source /opt/miniconda3/etc/profile.d/conda.sh && conda activate py39 && python --version"


    
# 安装VS Code服务器及扩展
RUN curl -fsSL https://code-server.dev/install.sh   | sh && \
    code-server --install-extension cnbcool.cnb-welcome && \
    code-server --install-extension redhat.vscode-yaml && \
    code-server --install-extension waderyan.gitblame && \
    code-server --install-extension mhutchie.git-graph && \
    code-server --install-extension donjayamanne.githistory && \
    code-server --install-extension cloudstudio.live-server && \
    code-server --install-extension tencent-cloud.coding-copilot@3.1.20 && \
    code-server --install-extension ms-python.debugpy && \
    code-server --install-extension ms-python.python




# 配置VS Code使用py39环境作为Python解释器
RUN mkdir -p /root/.config/code-server && \
    echo '{\n  "python.defaultInterpreterPath": "/opt/miniconda3/envs/py39/bin/python",\n  "python.condaPath": "/opt/miniconda3/bin/conda"\n}' > /root/.config/code-server/settings.json

# 设置字符集支持中文
ENV LANG=C.UTF-8
ENV LANGUAGE=C.UTF-8

# 验证Python版本（可选）
RUN python --version