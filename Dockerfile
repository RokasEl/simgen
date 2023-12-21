FROM pytorch/pytorch
EXPOSE 5000

RUN conda install -c anaconda git

# GIT CLONE: adapt if you are using an SSH key; remove once repo is public
COPY ./git-credentials /root/.git-credentials
RUN git config --global credential.helper store

RUN git clone https://github.com/RokasEl/MACE-Models
RUN git clone https://github.com/RokasEl/moldiff.git

# remove git credentials (I don't know if access to previous "RUN" is possible)
RUN rm /root/.git-credentials
# END GIT CLONE

WORKDIR /workspace/moldiff
RUN pip install .

WORKDIR /workspace/MACE-Models
RUN pip install .
RUN pip install --upgrade dvc-s3 dvc

RUN dvc remote modify origin --local access_key_id xxxx
RUN dvc remote modify origin --local secret_access_key xxxx
RUN dvc pull
RUN moldiff_init .

RUN pip install --upgrade git+https://github.com/zincware/zndraw@bd605b9530a35a37117ebb0dc9e333b843383338

COPY ./connect.py .

CMD moldiff_server --device cuda & python connect.py