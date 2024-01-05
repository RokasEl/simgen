FROM pytorch/pytorch
# EXPOSE 5000

RUN conda install -c anaconda git

RUN pip install git+https://github.com/ACEsuit/mace.git

COPY ./ /workspace/simgen
WORKDIR /workspace/simgen
RUN pip install .

RUN pip install --upgrade git+https://github.com/zincware/zndraw@main
RUN simgen init . --no-add-to-zndraw

CMD simgen connect --url "https://zndraw.icp.uni-stuttgart.de/" --device "cuda"
