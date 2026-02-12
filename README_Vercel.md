Deployment instructions — Vercel (frontend) + Render (recommended backend)

Summary
- Vercel is great for hosting static frontends. Large ML dependencies (PyTorch) often exceed Vercel serverless size limits.
- Recommended approach: deploy the API (Flask + PyTorch) to a server/host that supports large packages (Render, Railway, or a VPS), and deploy the frontend to Vercel as a static site.

Steps — Backend (Render recommended)
1. On Render (https://render.com) create a new Web Service.
2. Connect your GitHub repo and pick branch `main`.
3. Use the `requirements.txt` in the repo (it already includes the PyTorch CPU extra-index-url). Ensure `runtime.txt` is set to `python-3.11.9`.
4. Set the start command to: `gunicorn app.main:app --bind 0.0.0.0:$PORT` or use the default Render web service settings.
5. Deploy — Render will install dependencies including PyTorch CPU wheels and run the Flask app.

Steps — Frontend (Vercel)
1. In Vercel, create a new project and import this repo.
2. Vercel will read `vercel.json` and deploy the static frontend from `app/frontend`.
3. In Vercel project settings → Environment Variables, add `API_BASE_URL` set to your backend URL (for example `https://your-backend.onrender.com`).
4. Deploy. The frontend will call `${API_BASE_URL}/api/predict` and `${API_BASE_URL}/api/latest`.

Alternative (single-host):
- If you want both frontend and backend on one host, use Render to serve the Flask app (which serves both static files and the API). This avoids cross-origin setup.

Notes
- If you still want to attempt deploying the full app (including PyTorch) on Vercel, be aware of Lambda size limits and possible build failures. Vercel is not recommended for heavy ML inference.
