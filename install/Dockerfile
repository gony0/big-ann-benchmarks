FROM ubuntu:18.04

RUN apt-get update && apt-get install -y python3-numpy python3-scipy python3-pip build-essential git axel
RUN pip3 install -U pip

WORKDIR /home/app
COPY requirements.txt run_algorithm.py ./
RUN pip3 install -rrequirements.txt

ENTRYPOINT ["python3", "-u", "run_algorithm.py"]
