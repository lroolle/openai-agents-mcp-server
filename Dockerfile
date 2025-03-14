# Generated by https://smithery.ai. See: https://smithery.ai/docs/config#dockerfile
FROM python:3.11-slim

# set working directory
WORKDIR /app

# install build dependencies
RUN apt-get update && apt-get install -y gcc && rm -rf /var/lib/apt/lists/*

# copy project files
COPY . /app

# upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# install project dependencies using hatchling build system
RUN pip install --no-cache-dir .

# expose port if using SSE transport (optional)
EXPOSE 5173

# default command
CMD ["openai-agents-mcp-server"]
