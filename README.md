# The Garden of Forking Paths

An interactive narrative application using various language models.

## Environment Setup

1. Copy the environment template:
```bash
cp .env.example .env
```

2. Edit `.env` and add your API keys:
- `OPENAI_API_KEY`: Your OpenAI API key
- `ANTHROPIC_API_KEY`: Your Anthropic API key
- `OLLAMA_HOST`: URL for Ollama if using local models

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the application:
```bash
shiny run
```

## Docker Setup (Local)

1. Build the container:
```bash
docker-compose build
```

2. Run the application:
```bash
docker-compose up
```

The application will be available at http://localhost:8000

## Azure Deployment

### Prerequisites

1. Install Azure CLI: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli
2. Login to Azure:
```bash
az login
```

### Environment Variables Management

The deployment script handles environment variables securely:

1. Creates an Azure Key Vault
2. Stores all variables from your .env file as secrets
3. Configures the app to use Key Vault references
4. Sets up managed identity for secure access

To update environment variables after deployment:

1. Go to Azure Portal > Key Vault > garden-of-paths-vault
2. Navigate to Secrets
3. Add/Edit secrets as needed
4. The app will automatically use updated values

### Deployment Steps

1. Make the deployment script executable:
```bash
chmod +x deploy-azure.sh
```

2. Run the deployment script:
```bash
./deploy-azure.sh
```

### Setting Up Authentication

1. Go to Azure Portal > App Service > Authentication
2. Click "Add identity provider"
3. Select "Microsoft" as the identity provider
4. Configure the authentication:
   - Allow unauthenticated access: Off
   - Token store: On
   - Restrict access: Require authentication

### Managing Users

1. Go to Azure Portal > Azure Active Directory
2. Navigate to Users
3. Click "New user" to:
   - Create new users directly
   - OR invite external users (your friends)
4. For external users:
   - Click "New guest user"
   - Enter their email address
   - They'll receive an invitation email
   - Once accepted, they can access the app

### Security Best Practices

1. Environment Variables:
   - Stored in Azure Key Vault
   - Accessed via managed identity
   - No secrets in app settings
   - Rotatable without app changes

2. Authentication:
   - Azure AD integration
   - Secure token handling
   - User management through Azure Portal

3. Access Control:
   - Role-based access to Key Vault
   - Managed identity for app
   - No exposed credentials

### Monitoring & Management

Monitor your application in the Azure Portal:
1. Go to App Service > garden-of-paths
2. View logs, metrics, and configuration
3. Scale up/down as needed

View user access in Azure AD:
1. Go to Azure Active Directory > Users
2. Monitor user sign-ins
3. Manage access rights

Monitor secrets in Key Vault:
1. Go to Key Vault > garden-of-paths-vault
2. View access logs
3. Manage secret versions
4. Set up rotation policies

### Cost Management

The deployment uses:
- Basic (B1) tier App Service (~$13/month)
- Standard tier Key Vault (~$0.03/10,000 operations)
- Azure AD Free tier (included)
- Container Registry Basic tier (~$5/month)

You can:
- Scale down to Free tier for testing
- Scale up to Standard tier for better performance
- Enable auto-scaling for dynamic workloads
