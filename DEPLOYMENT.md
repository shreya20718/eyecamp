# Deployment Guide for Render

## Step-by-Step Deployment Instructions

### 1. Prepare Your Code

Make sure your project has these files:
- `package.json` ✅
- `server.js` ✅
- `public/index.html` ✅
- `render.yaml` ✅
- `.gitignore` ✅

### 2. Initialize Git Repository

```bash
# Initialize git repository
git init

# Add all files
git add .

# Make initial commit
git commit -m "Initial commit: PD Detector web app"
```

### 3. Push to GitHub

```bash
# Create a new repository on GitHub first, then:

# Add remote origin (replace with your GitHub repo URL)
git remote add origin https://github.com/yourusername/pd-detector.git

# Push to GitHub
git push -u origin main
```

### 4. Deploy to Render

#### Option A: Using Render Dashboard

1. Go to [render.com](https://render.com)
2. Sign up/Login with GitHub
3. Click **"New +"** → **"Web Service"**
4. Connect your GitHub repository
5. Configure the service:
   - **Name**: `pd-detector`
   - **Environment**: `Node`
   - **Build Command**: `npm install`
   - **Start Command**: `npm start`
   - **Plan**: Free
6. Click **"Create Web Service"**

#### Option B: Using render.yaml (Recommended)

1. Go to [render.com](https://render.com)
2. Sign up/Login with GitHub
3. Click **"New +"** → **"Blueprint"**
4. Connect your GitHub repository
5. Render will automatically detect `render.yaml`
6. Click **"Apply"**

### 5. Environment Variables

Render will automatically set:
- `NODE_ENV=production`
- `PORT=10000` (or similar)

### 6. Custom Domain (Optional)

1. In your Render dashboard
2. Go to your service settings
3. Add custom domain under **"Custom Domains"**
4. Update DNS records as instructed

## Commands Summary

```bash
# 1. Initialize and commit
git init
git add .
git commit -m "Initial commit: PD Detector web app"

# 2. Add GitHub remote (replace with your repo)
git remote add origin https://github.com/yourusername/pd-detector.git

# 3. Push to GitHub
git push -u origin main

# 4. Deploy on Render (via dashboard)
# - Connect GitHub repo
# - Use render.yaml configuration
# - Deploy automatically
```

## Post-Deployment

### Test Your App

1. Visit your Render URL: `https://your-app-name.onrender.com`
2. Test camera functionality
3. Verify all features work on mobile

### Update Your App

```bash
# Make changes to your code
git add .
git commit -m "Update: description of changes"
git push origin main

# Render will automatically redeploy
```

## Troubleshooting

### Common Issues

1. **Build fails**: Check `package.json` dependencies
2. **App crashes**: Check server logs in Render dashboard
3. **Camera not working**: Ensure HTTPS (Render provides this)
4. **Mobile issues**: Test on actual device, not just browser dev tools

### Render Dashboard

- **Logs**: View real-time logs
- **Metrics**: Monitor performance
- **Settings**: Update environment variables
- **Deploys**: View deployment history

## Cost

- **Free tier**: 750 hours/month
- **Paid plans**: Start at $7/month for always-on

## Support

- Render Documentation: https://render.com/docs
- Render Community: https://community.render.com
