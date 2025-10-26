# Biochar RAG System - Render Deployment Guide

## 🚀 Deployment Overview

This biochar RAG system is configured for deployment on Render with two separate services:
1. **API Backend** (FastAPI + RAG processing)
2. **Frontend** (Streamlit web interface)

## 📋 Pre-deployment Checklist

### 1. Data Files Required
Ensure these files are in your repository:
- `embeddings_output/biochar_embeddings.npy`
- `embeddings_output/biochar_chunks_with_embeddings.csv`
- `embeddings_output/embedding_metadata.json`

### 2. Environment Variables
Set up these environment variables in Render:
- `TOGETHER_API_KEY`: Your Together AI API key
- `API_URL`: URL of your deployed API service (for frontend)

## 🔧 Deployment Steps

### Option 1: Two-Service Deployment (Recommended)

#### Step 1: Deploy API Backend
1. Create new Web Service on Render
2. Connect your GitHub repository
3. Use configuration: `render-api.yaml`
4. Set environment variables:
   - `TOGETHER_API_KEY`: Your API key
   - Other variables are pre-configured
5. Deploy and note the service URL

#### Step 2: Deploy Frontend
1. Create another Web Service on Render
2. Connect the same GitHub repository
3. Use configuration: `render-frontend.yaml`
4. Update `API_URL` environment variable with your API service URL
5. Deploy

### Option 2: Single Service Deployment

Use the combined `render.yaml` configuration file.

## 🔍 Post-Deployment Verification

1. **API Health Check**: Visit `https://your-api-url.onrender.com/health`
2. **API Documentation**: Visit `https://your-api-url.onrender.com/docs`
3. **Frontend**: Visit your frontend URL and test the chat interface

## ⚙️ Configuration Files

- `render.yaml`: Combined deployment (both services)
- `render-api.yaml`: API backend only
- `render-frontend.yaml`: Frontend only
- `requirements.txt`: Python dependencies for API
- `streamlit-requirements.txt`: Minimal dependencies for frontend

## 🔧 Troubleshooting

### Common Issues:

1. **Missing embedding files**: Ensure all files in `embeddings_output/` are committed
2. **API connection failed**: Check `API_URL` environment variable in frontend
3. **Together AI errors**: Verify `TOGETHER_API_KEY` is set correctly
4. **Memory issues**: Consider upgrading to a higher Render plan

### Performance Optimization:

- Use CPU-optimized PyTorch builds
- Set `TOKENIZERS_PARALLELISM=false` to avoid warnings
- Consider using Render's Standard plan for better performance

## 📊 Resource Requirements

- **API Backend**: Minimum 512MB RAM (Starter plan)
- **Frontend**: 256MB RAM sufficient
- **Storage**: ~500MB for embeddings and models

## 🔐 Security Notes

- Password is hardcoded as "netzero" - consider using environment variables
- API key is properly secured through Render's environment variables
- CORS is enabled for cross-origin requests

## 📝 Updates and Maintenance

- Auto-deploy is enabled for both services
- Push to main branch triggers automatic redeployment
- Monitor logs through Render dashboard