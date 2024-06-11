FROM pytorch/pytorch
# EXPOSE 5000

RUN conda install -c anaconda git

RUN pip install git+https://github.com/ACEsuit/mace.git

COPY ./ /workspace/simgen
WORKDIR /workspace/simgen
RUN pip install -e .

RUN pip install -I git+https://github.com/zincware/zndraw@typescript


RUN git clone https://github.com/RokasEl/MACE-Models /workspace/MACE-Models
WORKDIR /workspace/MACE-Models
RUN git checkout develop

RUN pip uninstall hydromace -y
RUN pip install .
RUN pip install git+https://github.com/RokasEl/hydromace.git@develop
RUN dvc pull
# RUN simgen init . --no-add-to-zndraw


ENTRYPOINT [ "simgen", "connect" ]
