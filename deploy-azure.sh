#!/bin/bash

# Exit on error
set -e

echo "🚀 Deploying to Azure..."

# Login check
if ! az account show > /dev/null 2>&1; then
    echo "⚠️ Please login to Azure first using: az login"
    exit 1
fi

# Load configuration
LOCATION="westus"
BASE_NAME="gardenofpaths"
RESOURCE_GROUP="rg-$BASE_NAME"

echo "📦 Creating Resource Group..."
az group create --name $RESOURCE_GROUP --location $LOCATION

echo "🏗️ Deploying Infrastructure..."
az deployment group create \
    --resource-group $RESOURCE_GROUP \
    --template-file ./infra/main.bicep \
    --parameters @./infra/parameters.json

# Get resource names
ACR_NAME=$(az deployment group show -g $RESOURCE_GROUP -n main --query properties.outputs.containerRegistryName.value -o tsv)
APP_NAME=$(az deployment group show -g $RESOURCE_GROUP -n main --query properties.outputs.appServiceName.value -o tsv)
VAULT_NAME=$(az deployment group show -g $RESOURCE_GROUP -n main --query properties.outputs.keyVaultName.value -o tsv)

echo "🔑 Getting Registry Credentials..."
ACR_LOGIN_SERVER=$(az acr show --name $ACR_NAME --query loginServer --output tsv)
az acr login --name $ACR_NAME

echo "🏷️ Building and Pushing Docker Image..."
docker build -t $ACR_LOGIN_SERVER/$BASE_NAME:latest .
docker push $ACR_LOGIN_SERVER/$BASE_NAME:latest

# Store environment variables in Key Vault and configure app
if [ -f .env ]; then
    echo "📝 Setting Environment Variables in Key Vault..."
    while IFS='=' read -r key value; do
        if [ ! -z "$key" ] && [ ! -z "$value" ] && [[ ! $key =~ ^# ]]; then
            # Store in Key Vault
            az keyvault secret set --vault-name $VAULT_NAME \
                --name "${key}" \
                --value "${value}"
            
            # Configure app to use Key Vault reference
            az webapp config appsettings set \
                --resource-group $RESOURCE_GROUP \
                --name $APP_NAME \
                --settings "${key}=@Microsoft.KeyVault(SecretUri=https://$VAULT_NAME.vault.azure.net/secrets/${key}/)"
        fi
    done < .env
fi

echo "🔒 Setting up Azure AD Authentication..."
az webapp auth update \
    --resource-group $RESOURCE_GROUP \
    --name $APP_NAME \
    --enabled true \
    --action LoginWithAzureAD \
    --aad-token-issuer-url "https://login.microsoftonline.com/common"

echo "✅ Deployment Complete!"
echo "Your app should be available at: https://$APP_NAME.azurewebsites.net"
echo ""
echo "Environment variables are securely stored in Azure Key Vault: $VAULT_NAME"
echo ""
echo "Next steps:"
echo "1. Go to Azure Portal > App Service > Authentication"
echo "2. Set up Azure AD authentication"
echo "3. Invite users through Azure AD > Users"
