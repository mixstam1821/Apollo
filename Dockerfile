FROM python:3.11-slim

# System-level dependencies
RUN apt-get update && \
    apt-get install -y build-essential gcc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Bokeh app code
COPY Apollo.py .

# Expose default Bokeh port
EXPOSE 9651

# Run Bokeh server on container start
CMD ["bokeh", "serve", "Apollo.py", "--port=9651", "--allow-websocket-origin=*", "--allow-websocket-origin=localhost:9651"]
