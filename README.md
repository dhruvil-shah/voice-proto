# Solnce Voice Receptionist

A basic-authenticated Gradio voice receptionist for Solnce, a rooftop solar company in Gujarat. It accepts Gujarati, Hindi, or Indian English microphone audio, uses Gemini to answer, and returns Gemini-generated speech.

## Render deployment

1. Create a new Render Web Service from this GitHub repository.
2. Set:
   - Build command: `pip install -r requirements.txt`
   - Start command: `python app.py`
   - Instance type: Free
   - Python version: `3.10.13`
3. Add environment variables:
   - `GEMINI_API_KEY`: your Gemini API key
   - `APP_USER`: login username
   - `APP_PASS`: login password
   - `PYTHON_VERSION`: `3.10.13`

Render supplies `PORT` automatically. The app listens on `0.0.0.0` on that port.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set your Gemini key plus login credentials:

```dotenv
GEMINI_API_KEY=your-gemini-api-key
APP_USER=solnce
APP_PASS=choose-a-strong-password
```

Then start the app:

```bash
python app.py
```

Open `http://localhost:7860`, enter the basic-auth credentials, and use the microphone.

## Models

- Text/audio understanding: `gemini-3.8-flash`
- Text-to-speech: `gemini-3.8-flash-lite-tts`

The app uses Gemini REST APIs directly and does not expose the API key to the browser.
