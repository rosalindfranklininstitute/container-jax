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

FROM quay.io/rosalindfranklininstitute/jax:v0.3.1

ENV PATH="/usr/local/mpi/bin:/usr/local/mpi:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/mpi/lib64:/usr/local/mpi/lib:/usr/local/mpi:${LD_LIBRARY_PATH}"
ENV MPICC="/usr/local/mpi/bin/mpicc"
ENV MPI4JAX_USE_CUDA_MPI=1

# Copy compiled jax from the build image and install
COPY ./examples/baskerville/build-wheels/wheels /tmp/wheels
RUN pip install --no-cache-dir /tmp/wheels/mpi4py-3.1.3-cp38-cp38-linux_x86_64.whl && \
    pip install --no-cache-dir --no-build-isolation /tmp/wheels/mpi4jax-0.3.7-cp38-cp38-linux_x86_64.whl && \
    rm -rf /root/.cache/* &&  \
    rm -rf /tmp/* && \
    find /usr/lib/python3.*/ -name 'tests' -exec rm -rf '{}' +
WORKDIR /
