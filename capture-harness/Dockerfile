FROM alpine:3.14

WORKDIR "/harness"
COPY harness.py .
COPY requirements.txt .

RUN apk update && apk add tshark python3 py3-pip
RUN python3 -m pip install -r requirements.txt

CMD ["/usr/bin/python3", "/harness/harness.py"]
