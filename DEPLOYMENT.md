# Deployment Guide for Finance Agent

This guide provides instructions for deploying the Finance Agent application to various environments, from local development to cloud hosting.

## Table of Contents

- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [Cloud Hosting Options](#cloud-hosting-options)
  - [Heroku](#heroku)
  - [DigitalOcean App Platform](#digitalocean-app-platform)
  - [Railway](#railway)
  - [Render](#render)
  - [AWS Elastic Beanstalk](#aws-elastic-beanstalk)
  - [Google Cloud Run](#google-cloud-run)

## Local Development

1. Clone the repository:
   ```bash
   git clone https://github.com/shaunakc1998/finance-agent.git
   cd finance-agent
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env file to add your API keys
   ```

5. Run the application:
   ```bash
   python web/app.py
   ```

6. Access the application at http://localhost:5001

## Docker Deployment

### Using Docker Compose (Recommended for Local Deployment)

1. Clone the repository:
   ```bash
   git clone https://github.com/shaunakc1998/finance-agent.git
   cd finance-agent
   ```

2. Create a `.env` file with your API keys:
   ```bash
   cp .env.example .env
   # Edit .env file to add your API keys
   ```

3. Build and start the Docker container:
   ```bash
   docker-compose up -d
   ```

4. Access the application at http://localhost:5001

### Using Docker Directly

1. Clone the repository:
   ```bash
   git clone https://github.com/shaunakc1998/finance-agent.git
   cd finance-agent
   ```

2. Build the Docker image:
   ```bash
   docker build -t finance-agent .
   ```

3. Run the Docker container:
   ```bash
   docker run -p 5001:5001 \
     -e OPENAI_API_KEY=your_openai_api_key \
     -e ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key \
     -v $(pwd)/web/database:/app/web/database \
     -v $(pwd)/models:/app/models \
     -v $(pwd)/cache:/app/cache \
     finance-agent
   ```

4. Access the application at http://localhost:5001

## Cloud Hosting Options

### Heroku

1. Install the Heroku CLI:
   ```bash
   curl https://cli-assets.heroku.com/install.sh | sh
   ```

2. Login to Heroku:
   ```bash
   heroku login
   ```

3. Create a new Heroku app:
   ```bash
   heroku create finance-agent-app
   ```

4. Add a Procfile (already included in the repository):
   ```
   web: gunicorn -b 0.0.0.0:$PORT web.app:app
   ```

5. Set environment variables:
   ```bash
   heroku config:set OPENAI_API_KEY=your_openai_api_key
   heroku config:set ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
   ```

6. Deploy to Heroku:
   ```bash
   git push heroku main
   ```

7. Open the app:
   ```bash
   heroku open
   ```

### DigitalOcean App Platform

1. Create a DigitalOcean account if you don't have one.

2. Install the DigitalOcean CLI:
   ```bash
   curl -sL https://github.com/digitalocean/doctl/releases/download/v1.92.1/doctl-1.92.1-linux-amd64.tar.gz | tar -xzv
   sudo mv doctl /usr/local/bin
   ```

3. Authenticate with DigitalOcean:
   ```bash
   doctl auth init
   ```

4. Create a new app:
   ```bash
   doctl apps create --spec .do/app.yaml
   ```

5. Alternatively, use the DigitalOcean web interface:
   - Go to https://cloud.digitalocean.com/apps
   - Click "Create App"
   - Connect your GitHub repository
   - Configure environment variables
   - Deploy the app

### Railway

1. Create a Railway account at https://railway.app/

2. Install the Railway CLI:
   ```bash
   npm i -g @railway/cli
   ```

3. Login to Railway:
   ```bash
   railway login
   ```

4. Initialize a new project:
   ```bash
   railway init
   ```

5. Deploy the app:
   ```bash
   railway up
   ```

6. Set environment variables:
   ```bash
   railway variables set OPENAI_API_KEY=your_openai_api_key
   railway variables set ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
   ```

### Render

1. Create a Render account at https://render.com/

2. Create a new Web Service:
   - Connect your GitHub repository
   - Select "Python" as the runtime
   - Set the build command: `pip install -r requirements.txt`
   - Set the start command: `gunicorn web.app:app`
   - Add environment variables for your API keys

3. Deploy the app

### AWS Elastic Beanstalk

1. Install the AWS CLI and EB CLI:
   ```bash
   pip install awscli awsebcli
   ```

2. Configure AWS credentials:
   ```bash
   aws configure
   ```

3. Initialize Elastic Beanstalk:
   ```bash
   eb init
   ```

4. Create an environment:
   ```bash
   eb create finance-agent-env
   ```

5. Set environment variables:
   ```bash
   eb setenv OPENAI_API_KEY=your_openai_api_key ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
   ```

6. Deploy the app:
   ```bash
   eb deploy
   ```

### Google Cloud Run

1. Install the Google Cloud SDK:
   ```bash
   curl https://sdk.cloud.google.com | bash
   ```

2. Initialize the SDK:
   ```bash
   gcloud init
   ```

3. Build and push the Docker image:
   ```bash
   gcloud builds submit --tag gcr.io/your-project-id/finance-agent
   ```

4. Deploy to Cloud Run:
   ```bash
   gcloud run deploy finance-agent \
     --image gcr.io/your-project-id/finance-agent \
     --platform managed \
     --set-env-vars OPENAI_API_KEY=your_openai_api_key,ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
   ```

## Additional Configuration

### Persistent Storage

For production deployments, consider using a managed database service instead of SQLite:

1. Update the `DB_PATH` in `web/app.py` to use a PostgreSQL or MySQL connection string.
2. Add the appropriate database driver to `requirements.txt`.
3. Update the database initialization code to work with the new database.

### Security Considerations

1. Always use HTTPS in production.
2. Store API keys securely using environment variables or a secrets manager.
3. Implement proper authentication for the application.
4. Consider adding rate limiting to prevent abuse.

### Scaling

For high-traffic deployments:

1. Use a production-ready WSGI server like Gunicorn.
2. Consider using a load balancer for horizontal scaling.
3. Implement caching for API responses to reduce external API calls.
4. Use a CDN for static assets.
