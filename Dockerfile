# Use the official Python matching the general version used (e.g., 3.11 is safe)
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required for typical ML libraries and Folium
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the default Streamlit port (Render will override if needed)
EXPOSE 8501

# Command to run the application matching Render's deployment rules
CMD sh -c "streamlit run dashboard/app.py --server.port=${PORT:-10000} --server.address=0.0.0.0 --server.headless=true --server.enableCORS=false --server.enableWebsocketCompression=false"
