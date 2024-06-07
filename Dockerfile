FROM python:3.9-slim

WORKDIR /app

RUN pip install --no-cache-dir flask pandas numpy pymongo joblib scikit-learn mlflow onnxruntime dagshub  flask_cors requests
RUN mkdir -p src/serve

COPY src/serve/Predict_service.py src/serve/

ENV FLASK_APP=src/serve/Predict_service.py
ENV FLASK_RUN_HOST=0.0.0.0

EXPOSE 5000

CMD ["flask", "run"]