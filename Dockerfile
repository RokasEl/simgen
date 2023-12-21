FROM python:3.10-alpine
# install git
RUN apk add --no-cache git
# copy github credentials to container
COPY ./git-credentials /root/.git-credentials
RUN git config --global credential.helper store

RUN git clone https://github.com/RokasEl/MACE-Models
RUN git clone https://github.com/RokasEl/moldiff.git

# remove git credentials (I don't know if access to previous "RUN" is possible)
RUN rm /root/.git-credentials

RUN pip install torch
RUN pip install ./MACE-Models