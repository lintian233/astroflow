FROM quay.io/pypa/manylinux_2_28_x86_64
SHELL ["/bin/bash", "-lc"]

RUN dnf -y update && \
    dnf -y install wget curl xz bzip2 unzip git which findutils \
                   gcc gcc-c++ make patchelf cmake && \
    dnf clean all && rm -rf /var/cache/dnf

RUN dnf -y install dnf-plugins-core && \
    . /etc/os-release && \
    if [ "${VERSION_ID%%.*}" -ge 9 ]; then VER=rhel9; else VER=rhel8; fi && \
    dnf config-manager --add-repo \
      https://developer.download.nvidia.com/compute/cuda/repos/${VER}/x86_64/cuda-${VER}.repo && \
    dnf -y module disable nvidia-driver || true && \
    dnf -y install --setopt=install_weak_deps=False --nodocs \
      cuda-nvcc-12-9 \
      cuda-cudart-12-9 \
      cuda-cudart-devel-12-9 \
    && dnf clean all && rm -rf /var/cache/dnf

ENV CUDAHOME=/usr/local/cuda \
    CUDACXX=/usr/local/cuda/bin/nvcc \
    PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH} \
    NVCC_PREPEND_FLAGS=--allow-unsupported-compiler

RUN /opt/python/cp312-cp312/bin/pip install -U pip conan && \
    /opt/python/cp312-cp312/bin/conan --version && \
    /opt/python/cp312-cp312/bin/conan profile detect --force

RUN rm -f /tmp/depseed && mkdir -p /tmp/depseed
WORKDIR /tmp/depseed

COPY conanfile.txt /tmp/depseed/conanfile.txt

RUN /opt/python/cp312-cp312/bin/conan install . \
      -s build_type=Release \
      --build=missing

WORKDIR /root

RUN which gcc && gcc --version && \
    which g++ && g++ --version && \
    which nvcc && nvcc --version
