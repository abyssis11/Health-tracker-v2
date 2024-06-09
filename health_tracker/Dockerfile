FROM python:3.8-slim AS base

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . /app

# debugger

FROM base as debug

#RUN pip install debugpy

#RUN pip install ptvsd
RUN pip install debugpy
#CMD python -m ptvsd --host 0.0.0.0 --port 5678 --wait --multiprocess -m flask run -h 0.0.0 -p 5001

CMD python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m flask run -h 0.0.0.0 -p 5001

#ENTRYPOINT ["python", "-m", "debugpy", "--listen", "0.0.0.0:5678", "--wait-for-clinet", "-m"]

# primary

FROM base as production

CMD ["python", "app.py"]