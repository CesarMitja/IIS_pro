# Use an official lightweight Python image.
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install only the necessary packages.
RUN pip install --no-cache-dir flask pandas numpy pymongo joblib scikit-learn mlflow onnxruntime dagshub  flask_cors requests
# Create the expected directory structure as per your Flask app
RUN mkdir -p src/serve

# Copy only the necessary files
COPY src/serve/Predict_service.py src/serve/

# Set environment variables
ENV FLASK_APP=src/serve/Predict_service.py
ENV FLASK_RUN_HOST=0.0.0.0

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Run Flask when the container launches
CMD ["flask", "run"]