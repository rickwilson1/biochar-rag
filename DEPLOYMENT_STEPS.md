# 🚀 Deployment Instructions for Biochar RAG

## Step 1: Prepare Your Embeddings

### 1.1 Create a zip file of your embeddings
```bash
cd /Users/rickwilson/Documents/Python_Projects/biochar_rag
zip -r embeddings.zip embeddings_output/
```

### 1.2 Upload to Google Drive
1. Go to [Google Drive](https://drive.google.com)
2. Upload the `embeddings.zip` file
3. Right-click the file → "Get link" → "Anyone with the link"
4. Copy the sharing link (looks like: `https://drive.google.com/file/d/FILE_ID/view?usp=sharing`)
5. Extract the FILE_ID from the URL (the part between `/d/` and `/view`)

## Step 2: Commit Code to GitHub

```bash
# Add files to git
git add streamlit_app.py biochar_rag_api.py download_embeddings.py
git add requirements.txt streamlit-requirements.txt
git add render*.yaml .gitignore RENDER_DEPLOYMENT.md
git add .env.example

# Commit changes
git commit -m "Add Render deployment configuration with cloud storage support"

# Push to GitHub
git push origin main
```

## Step 3: Deploy API Backend on Render

1. Go to [Render Dashboard](https://dashboard.render.com)
2. Click "New" → "Web Service"
3. Connect your GitHub repository
4. Use these settings:
   - **Name**: `biochar-rag-api`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn biochar_rag_api:app --host 0.0.0.0 --port $PORT --workers 2 --worker-class uvicorn.workers.UvicornWorker`
5. Add Environment Variables:
   - `TOGETHER_API_KEY`: Your Together AI API key
   - `EMBEDDINGS_DRIVE_ID`: The FILE_ID from your Google Drive link
   - `TOKENIZERS_PARALLELISM`: `false`
6. Deploy and wait for it to finish
7. **Note the API URL** (e.g., `https://biochar-rag-api.onrender.com`)

## Step 4: Deploy Frontend on Render

1. Create another "Web Service"
2. Connect the same GitHub repository
3. Use these settings:
   - **Name**: `biochar-rag-frontend`
   - **Build Command**: `pip install streamlit requests`
   - **Start Command**: `streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`
4. Add Environment Variables:
   - `API_URL`: Your API URL from Step 3 (e.g., `https://biochar-rag-api.onrender.com`)
5. Deploy

## Step 5: Test Your Deployment

1. Visit your frontend URL
2. Enter password: `netzero`
3. Ask a biochar question
4. Verify it connects to your API and returns results

## 🔧 Troubleshooting

- **API won't start**: Check that `EMBEDDINGS_DRIVE_ID` is set correctly
- **Download fails**: Verify Google Drive link is public
- **Frontend can't connect**: Check `API_URL` environment variable
- **Out of memory**: Upgrade to Render Standard plan

## 📁 Files Created

- `download_embeddings.py` - Downloads embeddings from Google Drive
- `render-api.yaml` - API deployment configuration  
- `render-frontend.yaml` - Frontend deployment configuration
- Updated `biochar_rag_api.py` - Now downloads embeddings automatically
- Updated `streamlit_app.py` - Uses environment variable for API URL