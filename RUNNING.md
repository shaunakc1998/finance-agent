# Running the Finance Agent

This guide provides instructions for running the Finance Agent project locally, both with and without Docker.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Running Locally](#running-locally)
  - [Setting Up Environment Variables](#setting-up-environment-variables)
  - [Running the Web Interface](#running-the-web-interface)
  - [Running the CLI Chat Agent](#running-the-cli-chat-agent)
  - [Training Models](#training-models)
- [Running with Docker](#running-with-docker)
  - [Using Docker Compose](#using-docker-compose)
  - [Using Docker Directly](#using-docker-directly)
- [Troubleshooting](#troubleshooting)

## Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning the repository)
- Docker and Docker Compose (optional, for containerized deployment)

## Running Locally

### Setting Up Environment Variables

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key_here
   ```

3. You can get these API keys from:
   - OpenAI API Key: https://platform.openai.com/account/api-keys
   - Alpha Vantage: https://www.alphavantage.co/support/#api-key

### Running the Web Interface

1. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the web application:
   ```bash
   python web/app.py
   ```

4. Open your browser and navigate to http://localhost:5001

### Running the CLI Chat Agent

If you prefer using the command-line interface:

1. Ensure you have activated your virtual environment and installed dependencies as shown above.

2. Run the chat agent:
   ```bash
   python chat_agent.py
   ```

3. You can now interact with the finance agent through the command line.

Example commands:
- `forecast AAPL 10 days random_forest`
- `fundamentals NVDA`
- `technicals AMZN`
- `strategy TSLA`
- `financial insights AAPL What was the revenue growth?`

### Training Models

To train machine learning models for stock price forecasting:

1. Train a single model:
   ```bash
   python auto_train_model.py AAPL random_forest 10
   ```
   Arguments:
   - Ticker symbol (required)
   - Model type (optional, default: random_forest)
   - Forecast days (optional, default: 10)

2. Train multiple models:
   ```bash
   python run_training_background.py
   ```
   This will start the training process in the background and log the output to a file in the `training_logs` directory.

## Running with Docker

### Using Docker Compose

1. Ensure you have Docker and Docker Compose installed.

2. Create and configure your `.env` file as described in the [Setting Up Environment Variables](#setting-up-environment-variables) section.

3. Build and start the containers:
   ```bash
   docker-compose up -d
   ```

4. Open your browser and navigate to http://localhost:5001

5. To stop the containers:
   ```bash
   docker-compose down
   ```

### Using Docker Directly

1. Build the Docker image:
   ```bash
   docker build -t finance-agent .
   ```

2. Run the container:
   ```bash
   docker run -p 5001:5001 --env-file .env -d finance-agent
   ```

3. Open your browser and navigate to http://localhost:5001

4. To stop the container:
   ```bash
   docker stop $(docker ps -q --filter ancestor=finance-agent)
   ```

## Troubleshooting

### Common Issues

1. **API Key Issues**:
   - Ensure all required API keys are correctly set in your `.env` file.
   - Check that there are no extra spaces or quotes around your API keys.

2. **Port Already in Use**:
   - If port 5001 is already in use, you can modify the port in `web/app.py` or use a different port when running Docker.

3. **Python Version Compatibility**:
   - This project requires Python 3.8 or higher. Check your Python version with `python --version`.

4. **Package Installation Issues**:
   - If you encounter issues installing packages, try upgrading pip: `pip install --upgrade pip`
   - For specific package errors, check the package documentation or try installing the problematic package separately.

5. **Docker Issues**:
   - If Docker containers fail to start, check Docker logs: `docker-compose logs`
   - Ensure Docker and Docker Compose are up to date.

### Getting Help

If you encounter issues not covered here, please:
1. Check the existing issues on the GitHub repository
2. Create a new issue with detailed information about your problem
3. Include error messages, logs, and steps to reproduce the issue
