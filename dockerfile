# Dockerfile

FROM python:3.11.11

RUN git clone https://github.com/Matcap97/OBJECT_DETECTION.git .

WORKDIR /src

RUN pip3 install -r requirements.txt

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]