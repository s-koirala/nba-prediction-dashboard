# 🚀 Vercel Deployment Guide

Step-by-step guide to deploy the NBA Prediction Dashboard v3.0 to Vercel.

## Prerequisites

- GitHub account
- Vercel account (free tier is sufficient)
- Git installed locally
- Node.js 18+ installed

## Option 1: Deploy via GitHub (Recommended)

### Step 1: Create GitHub Repository

```bash
# Navigate to frontend directory
cd frontend-vercel

# Initialize git (if not already done)
git init

# Add files
git add .

# Commit
git commit -m "Initial commit: NBA Prediction Dashboard v3.0"

# Create GitHub repo at https://github.com/new
# Then push:
git remote add origin https://github.com/YOUR_USERNAME/nba-dashboard-v3.git
git branch -M main
git push -u origin main
```

### Step 2: Connect to Vercel

1. Go to [vercel.com](https://vercel.com)
2. Click "Add New..." > "Project"
3. Import your GitHub repository
4. Vercel auto-detects Next.js configuration
5. Click "Deploy"
6. Done! Your app is live in ~2 minutes

**Live URL**: `https://nba-dashboard-v3.vercel.app`

### Step 3: Configure Domain (Optional)

1. Go to Project Settings > Domains
2. Add custom domain (e.g., `nbapredictions.com`)
3. Follow DNS configuration instructions
4. SSL certificate auto-generated

## Option 2: Deploy via Vercel CLI

### Step 1: Install Vercel CLI

```bash
npm install -g vercel
```

### Step 2: Login to Vercel

```bash
vercel login
```

### Step 3: Deploy

```bash
# Navigate to frontend directory
cd frontend-vercel

# Deploy to preview
vercel

# Deploy to production
vercel --prod
```

### Step 4: Configure Environment Variables (if needed)

```bash
# Set environment variable
vercel env add PYTHON_API_URL

# Enter value when prompted
# Production: https://your-backend.herokuapp.com
```

## Option 3: Deploy from Dashboard

### Step 1: Build Locally

```bash
cd frontend-vercel
npm install
npm run build
```

### Step 2: Upload to Vercel

1. Go to [vercel.com/new](https://vercel.com/new)
2. Click "Deploy" tab
3. Drag and drop your project folder
4. Vercel builds and deploys
5. Done!

## Post-Deployment Configuration

### 1. Environment Variables

Set in Vercel Dashboard > Settings > Environment Variables:

```bash
PYTHON_API_URL=https://your-backend.com
NODE_ENV=production
```

### 2. Custom Domain

1. Project Settings > Domains
2. Add domain: `www.your-domain.com`
3. Configure DNS:
   ```
   Type: CNAME
   Name: www
   Value: cname.vercel-dns.com
   ```
4. Wait for DNS propagation (5-10 minutes)
5. SSL auto-configured

### 3. Analytics

Enable Vercel Analytics:
1. Project Settings > Analytics
2. Toggle "Enable"
3. View insights in Dashboard

### 4. Performance Monitoring

Enable Vercel Speed Insights:
1. Project Settings > Speed Insights
2. Toggle "Enable"
3. Monitor Core Web Vitals

## Connecting Python Backend

### Option A: Serverless Python (Experimental)

Create `api/python/predictions.py`:

```python
from http.server import BaseHTTPRequestHandler
import json
import pandas as pd

class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        predictions = pd.read_csv('results/tonights_predictions.csv')
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(predictions.to_dict('records')).encode())
```

Update `vercel.json`:
```json
{
  "functions": {
    "api/python/**/*.py": {
      "runtime": "python3.9"
    }
  }
}
```

### Option B: External API (Recommended)

Deploy Python backend separately:

**Heroku:**
```bash
# In main project directory
heroku create nba-predictions-api
git push heroku master
# URL: https://nba-predictions-api.herokuapp.com
```

**Railway:**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway init
railway up
# URL: https://nba-predictions-api.railway.app
```

**Render:**
1. Go to [render.com](https://render.com)
2. New > Web Service
3. Connect GitHub repo
4. Deploy
5. URL: `https://nba-predictions-api.onrender.com`

Update Vercel environment variable:
```bash
PYTHON_API_URL=https://nba-predictions-api.herokuapp.com
```

### Option C: File-Based (Simple)

Keep CSV in repository:

```bash
# Commit predictions CSV
git add results/tonights_predictions.csv
git commit -m "Update predictions"
git push

# Vercel auto-redeploys
```

**Note:** Vercel has 50MB total size limit for free tier.

## Updating Predictions

### Automatic Updates (Recommended)

Set up GitHub Action to run daily:

`.github/workflows/update-predictions.yml`:
```yaml
name: Update Predictions
on:
  schedule:
    - cron: '0 12 * * *'  # Run daily at noon UTC
  workflow_dispatch:

jobs:
  update:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install -r requirements.txt

      - name: Generate predictions
        run: |
          python predict_tonight_v2.py

      - name: Commit and push
        run: |
          git config user.name github-actions
          git config user.email github-actions@github.com
          git add results/tonights_predictions.csv
          git commit -m "Update predictions $(date)"
          git push
```

Vercel auto-redeploys on push.

### Manual Updates

```bash
# Generate predictions locally
python predict_tonight_v2.py

# Commit and push
cd frontend-vercel
git add ../results/tonights_predictions.csv
git commit -m "Update predictions"
git push

# Vercel auto-redeploys
```

## Monitoring & Maintenance

### 1. Check Deployment Status

```bash
# Via CLI
vercel ls

# Or Dashboard
https://vercel.com/YOUR_USERNAME/nba-dashboard-v3
```

### 2. View Logs

```bash
# Real-time logs
vercel logs

# Or Dashboard > Deployments > Logs
```

### 3. Rollback if Needed

```bash
# List deployments
vercel ls

# Rollback to previous deployment
vercel rollback DEPLOYMENT_URL
```

### 4. Monitor Performance

Dashboard > Analytics:
- Page views
- Unique visitors
- Top pages
- Device breakdown
- Geographic distribution

Dashboard > Speed Insights:
- Core Web Vitals
- Lighthouse scores
- Performance metrics

## Troubleshooting

### Build Fails

**Error:** `Module not found: Can't resolve 'X'`

**Solution:**
```bash
# Clear cache and reinstall
rm -rf node_modules .next
npm install
npm run build
```

### API Not Working

**Error:** `Failed to fetch predictions`

**Solutions:**
1. Check environment variable `PYTHON_API_URL` is set
2. Verify backend is running: `curl https://your-backend.com/api/predictions`
3. Check CORS headers in backend
4. View logs: `vercel logs --follow`

### Deployment Limit Exceeded

**Error:** `Deployment size exceeds limit`

**Solutions:**
1. Remove large files from repository
2. Add to `.gitignore`: `*.csv`, `*.pkl` (large model files)
3. Use external storage for models (S3, Google Cloud Storage)
4. Upgrade to Vercel Pro (250MB limit)

### Slow Performance

**Solutions:**
1. Enable ISR (Incremental Static Regeneration)
2. Implement caching in API routes
3. Optimize images with `next/image`
4. Use Edge Functions for API routes
5. Enable Vercel Analytics to identify bottlenecks

## Security Checklist

- [ ] Environment variables set (no secrets in code)
- [ ] HTTPS enabled (automatic on Vercel)
- [ ] Security headers configured (see `vercel.json`)
- [ ] CORS configured for production domains only
- [ ] Rate limiting on API routes
- [ ] Input validation and sanitization
- [ ] Authentication (if needed)
- [ ] Regular dependency updates (`npm audit`)

## Performance Checklist

- [ ] Images optimized (WebP, compressed)
- [ ] Fonts preloaded
- [ ] Lazy loading implemented
- [ ] Code splitting enabled (automatic in Next.js)
- [ ] Bundle analyzed: `npm run build -- --analyze`
- [ ] Lighthouse score >90 (mobile)
- [ ] Core Web Vitals green
- [ ] CDN enabled (automatic on Vercel)

## SEO Checklist

- [ ] Meta tags configured (see `app/layout.tsx`)
- [ ] Open Graph tags added
- [ ] Sitemap generated
- [ ] robots.txt configured
- [ ] Canonical URLs set
- [ ] Structured data (JSON-LD)
- [ ] Mobile-friendly (tested)
- [ ] Fast loading (<3s)

## Cost Estimation

### Vercel Free Tier
- ✅ 100GB bandwidth/month
- ✅ Unlimited deployments
- ✅ Automatic HTTPS
- ✅ CDN
- ✅ Analytics (limited)
- ✅ 1 concurrent build

**Cost:** $0/month

### Vercel Pro ($20/month)
- ✅ 1TB bandwidth
- ✅ Priority support
- ✅ Advanced analytics
- ✅ 12 concurrent builds
- ✅ Password protection
- ✅ Team collaboration

**Recommended for:** Production apps with >10k visitors/month

### Vercel Enterprise (Custom)
- ✅ Custom bandwidth
- ✅ 99.99% SLA
- ✅ Dedicated support
- ✅ SSO
- ✅ Advanced security

**Recommended for:** Large-scale production apps

## Next Steps

After successful deployment:

1. **Test thoroughly**
   - Mobile devices (iOS, Android)
   - Multiple browsers (Chrome, Safari, Firefox)
   - Different screen sizes
   - Network conditions (3G, 4G, WiFi)

2. **Monitor performance**
   - Vercel Analytics
   - Google Analytics
   - Error tracking (Sentry)
   - Uptime monitoring (UptimeRobot)

3. **Iterate based on data**
   - User feedback
   - Analytics insights
   - Performance metrics
   - Conversion rates

4. **Scale as needed**
   - Upgrade Vercel plan
   - Optimize backend
   - Add caching layers
   - Implement CDN

## Support

- **Vercel Documentation**: https://vercel.com/docs
- **Next.js Documentation**: https://nextjs.org/docs
- **Community Discord**: https://discord.gg/vercel
- **GitHub Issues**: https://github.com/YOUR_USERNAME/nba-dashboard-v3/issues

## Useful Commands

```bash
# Deploy preview
vercel

# Deploy production
vercel --prod

# View logs
vercel logs

# List deployments
vercel ls

# Remove deployment
vercel rm DEPLOYMENT_URL

# Check project status
vercel inspect

# Pull environment variables
vercel env pull

# Add domain
vercel domains add example.com

# Check DNS
vercel dns ls
```

---

**Ready to deploy?** Run `vercel` in the `frontend-vercel` directory!

**Questions?** Check the [README.md](README.md) or Vercel documentation.

**Good luck! 🚀**
