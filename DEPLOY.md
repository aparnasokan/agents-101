# Deploying Agent 101

This app is ready to deploy as a FastAPI web service.

## Replit

Use the FastAPI app in `main.py`, not `agent101-demo.html`.

Why:
- `main.py` keeps your Azure OpenAI key on the server.
- `agent101-demo.html` contains browser-side API key wiring and is not suitable for a public deployment.

### 1. Import the project into Replit

- Create a new Repl from this repo or upload the folder.

### 2. Add secrets

In the Replit Secrets panel, add:

```env
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=your-deployment-name
AZURE_OPENAI_API_VERSION=2024-10-21
AUTO_OPEN_BROWSER=false
```

Notes:
- `AUTO_OPEN_BROWSER=false` is recommended for hosted environments.
- Replit provides `PORT` automatically during deployment. `main.py` now respects that.

### 3. Run locally inside Replit

Use:

```bash
python main.py
```

The app will bind to Replit's `PORT` when present, otherwise it falls back to `8000`.

### 4. Deploy

- Click `Deploy`.
- Choose `Autoscale` for a lighter-weight public demo, or `Reserved VM` if you want the app to stay warm for presentations.
- Set the run command to:

```bash
python main.py
```

### 5. Protect access

If you want only invited people to use it:
- Turn on Replit's deployment protection.
- Choose either password protection or restricted/private access, depending on your Replit plan and deployment settings.

### 6. Share the URL

Your users can then visit a URL like:

```text
https://your-app-name.replit.app
```

## Other hosts

This app also works well on Render, Railway, Fly.io, or Azure App Service.

Suggested start command:

```bash
uvicorn main:app --host 0.0.0.0 --port $PORT
```

Required Python dependencies are listed in `requirements.txt`.
