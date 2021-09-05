FROM ubuntu

ARG CACHEBUST=19

ENV LC_ALL C.UTF-8

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get clean \
 && apt-get update --fix-missing \
 && apt-get install -y \
    python3 \
    python3-pip

WORKDIR /work

# install requirements
COPY ./Algos/Chapter-ID/requirements.txt /work
RUN pip3 install --upgrade setuptools
RUN pip3 install --no-cache-dir -r requirements.txt

COPY ./Chapter-Name.py /work
ADD ./Algos/utils /work/base/utils

RUN mkdir -p /work/.streamlit
COPY ./.streamlit /work/.streamlit

ADD ./Algos/Chapter-ID /work

RUN find /work -name '*.py' | xargs sed -i 's/Algos.utils/base.utils/g'
RUN find /work -name '*.py' | xargs sed -i 's/Algos.Chapter-ID.//g'

ENTRYPOINT ["streamlit", "run"]

CMD ["Chapter-Name.py"]
