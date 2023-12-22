FROM pytorch/pytorch
# EXPOSE 5000

RUN conda install -c anaconda git

# GIT CLONE: adapt if you are using an SSH key; remove once repo is public
COPY ./git-credentials /root/.git-credentials
RUN git config --global credential.helper store

RUN git clone https://github.com/RokasEl/MACE-Models
# RUN git clone https://github.com/RokasEl/moldiff.git
COPY ./ /workspace/moldiff

# remove git credentials (I don't know if access to previous "RUN" is possible)
RUN rm /root/.git-credentials
# END GIT CLONE

WORKDIR /workspace/moldiff
RUN pip install .

WORKDIR /workspace/MACE-Models
RUN pip install .
RUN pip install --upgrade dvc-s3 dvc

RUN dvc remote modify origin --local access_key_id 04278cea04be32b7479600c1d5d76b2d61d1a2a7
RUN dvc remote modify origin --local secret_access_key 04278cea04be32b7479600c1d5d76b2d61d1a2a7
RUN dvc pull
RUN moldiff_init .

RUN pip install --upgrade git+https://github.com/zincware/zndraw@main
COPY ./connect.py .
CMD moldiff_server --device cuda & python connect.py