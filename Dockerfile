
# build docker image:
# docker build . -t kizervus

# run docker container:
# docker run -p 80:80 -it kizervus

FROM python:3.8

ADD ./deployment/requirements.txt ./
ADD ./deployment/app.py ./
ADD ./deployment/model.onnx ./
ADD ./training/models/latest ./

RUN pip install -r requirements.txt

CMD ["python",  "app.py"]
