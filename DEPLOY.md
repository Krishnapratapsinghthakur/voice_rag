# 🚀 Railway Deployment Guide

Deploy the Voice AI Assistant to Railway with these steps.

## Prerequisites

1. A [Railway](https://railway.app) account (free tier available)
2. Your project pushed to GitHub
3. Google API Key for Gemini

## Architecture

```
┌─────────────────────┐     ┌─────────────────────┐
│   Frontend (React)  │────▶│   Backend (FastAPI) │
│   Railway Service   │     │   Railway Service   │
│   Port: 3000        │     │   Port: 8000        │
└─────────────────────┘     └─────────────────────┘
                                      │
                                      ▼
                            ┌─────────────────────┐
                            │   ChromaDB (local)  │
                            │   + Gemini API      │
                            └─────────────────────┘
```

## Step 1: Push to GitHub

```bash
# Initialize git if not already
git init
git add .
git commit -m "Prepare for Railway deployment"

# Push to GitHub
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

## Step 2: Deploy Backend

1. Go to [railway.app](https://railway.app) and login
2. Click **"New Project"** → **"Deploy from GitHub repo"**
3. Select your repository
4. Railway will detect the `railway.toml` file

### Configure Backend Environment Variables:

In Railway dashboard → Variables, add:

| Variable | Value |
|----------|-------|
| `GOOGLE_API_KEY` | Your Gemini API key |
| `FRONTEND_URL` | `https://your-frontend.railway.app` (add after frontend deploy) |

5. Wait for deployment to complete
6. Copy your backend URL (e.g., `https://voice-ai-backend-xxx.railway.app`)

## Step 3: Deploy Frontend

1. In Railway, click **"New"** → **"GitHub Repo"**
2. Select the same repo but configure as follows:
   - **Root Directory**: `frontend`
   - This tells Railway to deploy from the frontend folder

### Configure Frontend Environment Variables:

| Variable | Value |
|----------|-------|
| `VITE_API_URL` | Your backend URL from Step 2 (e.g., `https://voice-ai-backend-xxx.railway.app`) |

3. Wait for deployment to complete
4. Copy your frontend URL

## Step 4: Update Backend CORS

Go back to your **Backend** service in Railway:

1. Add/Update environment variable:
   - `FRONTEND_URL` = Your frontend URL (e.g., `https://voice-ai-frontend-xxx.railway.app`)

2. Railway will automatically redeploy

## Step 5: Verify Deployment

1. Open your frontend URL in browser
2. Try the Voice Assistant
3. Test the Admin Panel file upload

## Troubleshooting

### Backend not starting
- Check logs in Railway dashboard
- Verify `GOOGLE_API_KEY` is set correctly

### CORS errors
- Make sure `FRONTEND_URL` is set in backend
- Check if URLs match exactly (no trailing slashes)

### ChromaDB issues
- First upload creates the database
- Data persists on Railway's ephemeral storage (may reset on redeploy)

## For Persistent Storage

For production, consider:
1. **Railway Volumes** - Add persistent storage for ChromaDB
2. **Pinecone/Weaviate** - Use a managed vector database

## Costs

Railway Free Tier includes:
- $5/month credit
- Enough for small projects
- Sleeps after inactivity (wakes on request)

## Files Created for Deployment

- `railway.toml` - Backend deployment config
- `Procfile` - Process configuration
- `frontend/railway.toml` - Frontend deployment config
- `frontend/.env.example` - Environment template
