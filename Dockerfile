FROM nvcr.io/nvidia/pytorch:24.03-py3

# Install requirements
# COPY requirements.txt /app/requirements.txt
# WORKDIR /app
# RUN pip install -r requirements.txt

# Copy the Flask app
# COPY . /app

# Run the Flask app
# CMD ["python", "app.py"]

# Install dependencies
# WORKDIR /app
# COPY requirements.txt /app/
# RUN pip install -r requirements.txt

# Copy the entrypoint script and make it executable
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# Set the entrypoint script
ENTRYPOINT ["entrypoint.sh"]
CMD ["python", "predict.py"]
# CMD /bin/bash