# docker/manylinux_2_28_cuda129_x86_64/Dockerfile
FROM quay.io/pypa/manylinux_2_28_x86_64

# 用 bash 作为 shell，方便后续操作
SHELL ["/bin/bash", "-lc"]

# 基础工具 + 启用 gcc-toolset:12 模块 + 安装对应子包
RUN dnf -y update && \
    dnf -y install \
      wget curl xz bzip2 unzip git which findutils \
      gcc gcc-c++ make patchelf cmake && \
    dnf clean all && rm -rf /var/cache/dnf

# 如需改版本，只要覆盖 CUDA_URL 即可
ARG CUDA_URL="https://developer.download.nvidia.com/compute/cuda/12.9.0/local_installers/cuda_12.9.0_575.51.03_linux.run"
RUN curl -fSL "${CUDA_URL}" -o /tmp/cuda.run \
 && sh /tmp/cuda.run --silent --toolkit --no-man-page --override \
 && rm -f /tmp/cuda.run

ENV CUDAHOME=/usr/local/cuda \
    CUDACXX=/usr/local/cuda/bin/nvcc \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

RUN if [ -x /opt/python/cp312-cp312/bin/pip ]; then \
      /opt/python/cp312-cp312/bin/pip install -U pip conan && \
      /opt/python/cp312-cp312/bin/conan --version && \
      /opt/python/cp312-cp312/bin/conan profile detect --force; \
    fi


RUN which gcc && gcc --version && \
    which g++ && g++ --version && \
    which nvcc && nvcc --version
