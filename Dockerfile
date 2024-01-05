FROM pytorch/pytorch
# EXPOSE 5000

RUN conda install -c anaconda git

RUN pip install git+https://github.com/ACEsuit/mace.git
RUN git clone https://github.com/RokasEl/MACE-Models
WORKDIR /workspace/MACE-Models
RUN dvc pull

COPY ./ /workspace/simgen
WORKDIR /workspace/simgen
RUN pip install .
RUN pip install --upgrade git+https://github.com/zincware/zndraw@main

WORKDIR /workspace/MACE-Models
RUN simgen init . --no-add-to-zndraw

CMD simgen connect --url "https://zndraw.icp.uni-stuttgart.de/" \
    --device "cuda" \
    --mace-model-name "medium_spice" \
    --reference-data-name "simgen_reference_data_medium"
