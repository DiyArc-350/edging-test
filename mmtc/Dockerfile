FROM python:3.9-slim

WORKDIR /app

# Ensure Python output is sent straight to terminal (e.g. your container log)
# without being first buffered and that you can see the output of your application
# in real time.
ENV PYTHONUNBUFFERED=1

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the port the app runs on
EXPOSE 8080

CMD ["python", "main.py"]
