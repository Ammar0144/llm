# Deployment Configuration

## Server Requirements

- Docker installed
- Ubuntu/Debian Linux (recommended)
- At least 2GB RAM
- At least 5GB storage space
- Port 8082 available

## Required GitHub Secrets

Set these in your GitHub repository settings (Settings > Secrets and variables > Actions):

### Required Secrets:
- `SERVER_HOST`: Your server's IP address or domain name
- `SERVER_USER`: SSH username (e.g., "ubuntu", "root")
- `SERVER_SSH_KEY`: Your private SSH key content

### Optional Secrets:
- `SERVER_PORT`: SSH port (default)

## Deployment Process

1. **Push to main branch** → Triggers CI/CD pipeline
2. **CI runs tests** → Builds Docker image
3. **Deploy job** → Copies image to server and starts container

## Manual Deployment

If you prefer to deploy manually:

```bash
# On your server
cd ~/llm-deploy
wget https://github.com/Ammar0144/llm/releases/latest/download/llm-server.tar
docker load -i llm-server.tar
docker stop llm-server || true
docker rm llm-server || true
docker run -d --name llm-server --restart unless-stopped -p 8082:8082 llm-server:latest
```

## Health Check

After deployment, verify the service:

```bash
curl http://your-server:8082/health
```

## Logs

View container logs:

```bash
docker logs llm-server -f
```

## Rollback

To rollback to previous version:

```bash
docker stop llm-server
docker rm llm-server
docker run -d --name llm-server --restart unless-stopped -p 8082:8082 llm-server:old
```
