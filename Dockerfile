FROM hub-dev.hexin.cn/jupyterhub/nvidia_cuda:py37-cuda100-ubuntu18.04-v2

COPY ./ /home/jovyan/table-recognition 

RUN cd /home/jovyan/table-recognition  && \
    python -m pip install -r requirement.txt 