FROM tensorflow/tensorflow:1.6.0-gpu

# Setup environment for tensorflow models
RUN apt-get update && apt-get install -y \
        build-essential \
        curl \
        git \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python-dev \
        python-numpy \
        python-pip \
        protobuf-compiler \
        python-pil \
        python-lxml \
        python-tk \
        software-properties-common \
        swig \
        zip \
        zlib1g-dev \
        libcurl3-dev \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get install python-pip python-dev build-essential

#RUN curl -fSsL -O https://bootstrap.pypa.io/get-pip.py && \
#    python get-pip.py && \
#    rm get-pip.py

# Set up grpc
RUN pip install enum34 futures mock six matplotlib jupyter && \
    pip install --pre 'protobuf>=3.0.0a3' && \
    pip install -i https://testpypi.python.org/simple --pre grpcio

WORKDIR /
RUN git clone -b r1.4 https://github.com/tensorflow/tensorflow.git

# Set up Bazel.

# Running bazel inside a `docker build` command causes trouble, cf:
#   https://github.com/bazelbuild/bazel/issues/134
# The easiest solution is to set up a bazelrc file forcing --batch.
RUN echo "startup --batch" >>/etc/bazel.bazelrc
# Similarly, we need to workaround sandboxing issues:
#   https://github.com/bazelbuild/bazel/issues/418
RUN echo "build --spawn_strategy=standalone --genrule_strategy=standalone" \
    >>/etc/bazel.bazelrc
# Install the most recent bazel release.
ENV BAZEL_VERSION 0.5.4
WORKDIR /
RUN mkdir /bazel && \
    cd /bazel && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -O https://github.com/bazelbuild/bazel/releases/download/$BAZEL_VERSION/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    curl -H "User-Agent: Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2987.133 Safari/537.36" -fSsL -o /bazel/LICENSE.txt https://raw.githubusercontent.com/bazelbuild/bazel/master/LICENSE && \
    chmod +x bazel-*.sh && \
    ./bazel-$BAZEL_VERSION-installer-linux-x86_64.sh && \
    cd / && \
    rm -f /bazel/bazel-$BAZEL_VERSION-installer-linux-x86_64.sh

WORKDIR /tensorflow
RUN tensorflow/tools/ci_build/builds/configured CPU \
	bazel build tensorflow/tools/graph_transforms:transform_graph

WORKDIR /
RUN git clone https://github.com/tensorflow/models.git
RUN git clone https://github.com/cocodataset/cocoapi.git
WORKDIR /cocoapi/PythonAPI 
RUN pip install Cython && make
RUN cp -r pycocotools /models/research/

WORKDIR /models/research/
RUN protoc object_detection/protos/*.proto --python_out=.
ENV PYTHONPATH=$PYTHONPATH:/models/research/:/models/research/slim

# Add dataset into docker file
# Add volumes for output graph and 
VOLUME /data

# Add files to dockerfile
ADD . /
RUN chmod +x /run.sh

EXPOSE 8000
ENTRYPOINT ["/run.sh"]
CMD [""]