# Dockerfile

FROM ultralytics/ultralytics:latest-cpu

EXPOSE 8080

COPY . .

RUN pip install -U pip
RUN pip install -r requirements.txt

HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health

ENTRYPOINT ["streamlit", "run", "./src/streamlit_app.py", "--server.port=8080"]