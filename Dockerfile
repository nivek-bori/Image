FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy your code
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Run Command
CMD ['python', 'eval.py', 'w', '1']