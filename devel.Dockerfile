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

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as build

# Install packages and register python3 as python
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections && \
    apt-get update -y && apt-get install --no-install-recommends -y dialog apt-utils && \
    apt-get install --no-install-recommends -y \
      g++ git wget cython3 python3 python3-dev python3-pip && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 10 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 10 && \
    apt-get autoremove -y --purge && apt-get clean -y && rm -rf /var/lib/apt/lists/*

# Install python packages
RUN pip install --no-cache-dir --upgrade \
        numpy>=1.22.2 six wheel mock pytest pytest-cov PyYAML coverage

RUN pip install --upgrade "jax[cuda11_local]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN pip install --no-cache-dir --upgrade \
        h5py scipy>=1.8.0 scikit-image>=0.19.2 imageio tqdm pandas matplotlib && \
    pip install --no-cache-dir --upgrade \
        git+git://github.com/deepmind/dm-haiku.git@f25eb03a959d26c8ca97eca13cc8ca4678dd3967 && \
    pip install --no-cache-dir --upgrade \
        optax jupyter jupyterlab && \
    rm -rf /root/.cache/* &&  \
    rm -rf /tmp/* && \
    find /usr/lib/python3.*/ -name 'tests' -exec rm -rf '{}' +
WORKDIR/
