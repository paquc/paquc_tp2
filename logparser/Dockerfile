FROM python:3.8-slim

ADD requirements.txt .
ADD setup.py .

RUN pip3 install -r requirements.txt

CMD ["python3", "--version"]
