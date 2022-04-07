# Copyright 2022 Rosalind Franklin Institute
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
# either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

FROM nvidia/cuda:11.5.1-cudnn8-devel-ubuntu20.04 as build
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64:/usr/local/cuda/compat/"

# Install packages and register python3 as python
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections && \
    apt-get update -y && apt-get install --no-install-recommends -y dialog apt-utils && \
    apt-get install --no-install-recommends -y \
      g++ git wget python cython3 python3 python3-dev python3-pip && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 10 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10 && \
    apt-get autoremove -y --purge && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install python packages
RUN pip install --no-cache-dir --upgrade \
    numpy>=1.22.2 six wheel mock pytest pytest-cov PyYAML coverage

# Install jax from source
WORKDIR /usr/local/jax
RUN git clone --branch jax-v0.3.1 --depth 1 https://github.com/google/jax.git . && \
    python build/build.py  \
        --enable_cuda \
        --cuda_path='/usr/local/cuda' \
        --cudnn_path='/usr' \
        --cuda_version='11.5' \
        --cudnn_version='8' && \
    pip install --no-cache-dir --upgrade dist/*.whl && \
    pip install -e . && \
    rm -rf /root/.cache/* &&  \
    rm -rf /tmp/* && \
    find /usr/lib/python3.*/ -name 'tests' -exec rm -rf '{}' +

FROM nvidia/cuda:11.5.1-cudnn8-runtime-ubuntu20.04
ENV LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:/usr/local/cuda/lib64:/usr/local/cuda/compat/"

# Install runtime packages
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections && \
    apt-get update -y && apt-get install --no-install-recommends -y dialog apt-utils && \
    apt-get install --no-install-recommends -y python cython3 python3 python3-pip && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 10 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10 && \
    apt-get autoremove -y --purge && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install python packages
RUN pip install --no-cache-dir --upgrade \
    numpy>=1.22.2 six wheel mock pytest pytest-cov PyYAML coverage

# Copy compiled jax from the build image and install
COPY --from=build /usr/local/jax /usr/local/jax
COPY --from=build /usr/local/cuda/bin/ptxas /usr/local/cuda/bin/ptxas
WORKDIR /usr/local/jax
RUN pip install --no-cache-dir --upgrade dist/*.whl && \
    rm -rf dist/*.whl && \
    pip install -e . && \
    rm -rf /root/.cache/* &&  \
    rm -rf /tmp/* && \
    find /usr/lib/python3.*/ -name 'tests' -exec rm -rf '{}' +
WORKDIR /
