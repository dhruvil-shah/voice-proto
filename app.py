import base64
import os
import tempfile
import time
import wave

import gradio as gr
import httpx
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("GEMINI_API_KEY")
APP_USER = os.environ.get("APP_USER")
APP_PASS = os.environ.get("APP_PASS")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
if not APP_USER or not APP_PASS:
    raise RuntimeError("APP_USER and APP_PASS environment variables must be set.")

TEXT_MODEL = "gemini-3.5-flash"
TEXT_FALLBACK_MODEL = "gemini-3.1-flash-lite"
TTS_MODEL = "gemini-3.8-flash-lite-tts"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

SYSTEM_PROMPT = """
You are a warm, real human phone receptionist for 'Solnce' (pronounced Sol-anse), a rooftop solar company in Gujarat.

CRITICAL VOICE & PRONUNCIATION RULES:
1. Match the caller's language (Gujarati, Hindi, or Indian English).
2. NEVER output digits (0-9), decimals (2.3), currency symbols (₹, Rs), or abbreviations (kW, sq ft).
3. ALWAYS write every number, decimal, and unit out in full spoken words in that language's script:
   - Instead of "2.3 Lakh", write "બે પોઈન્ટ ત્રણ લાખ" (Gujarati) or "दो पॉइंट तीन लाख" (Hindi) or "two point three lakh" (English).
   - Instead of "₹78,000", write "અઠ્યોતેર હજાર રૂપિયા" (Gujarati) or "अठहत्तर हज़ार रुपये" (Hindi).
   - Instead of "3 kW", write "ત્રણ કિલોવોટ" (Gujarati) or "तीन किलोवाट" (Hindi).
   - Write "Solnce" phonetically as "સોલંસ" in Gujarati and "सोलंस" in Hindi.
4. Sound like a real person on a phone call: use natural conversational openers ("જી હા", "બિલકુલ", "अच्छा जी", "Sure!") and keep answers to 1-2 short, natural sentences.
5. Solar facts: 1 kilowatt generates 120 units a month (~900 rupees bill) and costs around 55 thousand rupees before the PM Surya Ghar subsidy (up to 78 thousand rupees subsidy for 3 kilowatt and above).
"""


def api_request(model, payload, timeout=35.0):
    url = GEMINI_URL.format(model=model)
    with httpx.Client(timeout=timeout) as client:
        response = client.post(
            url,
            headers={"x-goog-api-key": API_KEY, "Content-Type": "application/json"},
            json=payload,
        )
        response.raise_for_status()
        return response.json()


def extract_text(data):
    for candidate in data.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if "text" in part:
                return part["text"].strip()
    return ""


def extract_audio(data):
    for candidate in data.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                return base64.b64decode(inline["data"]), inline.get("mimeType") or inline.get("mime_type")
    return None, None


def save_wav(audio_bytes, mime_type=None):
    if audio_bytes.startswith(b"RIFF") and audio_bytes.find(b"WAVE") != -1:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(audio_bytes)
            return tmp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        with wave.open(tmp.name, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(24000)
            wav_file.writeframes(audio_bytes)
        return tmp.name


def process_voice(audio_filepath, chat_ui, history):
    if not audio_filepath:
        return None, chat_ui or [], history or []

    with open(audio_filepath, "rb") as audio_file:
        wav_bytes = audio_file.read()

    current_history = history or []
    current_history.append({
        "role": "user",
        "parts": [{"inlineData": {"mimeType": "audio/wav", "data": base64.b64encode(wav_bytes).decode("ascii")}}],
    })

    text_payload = {
        "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
        "contents": current_history[-8:],
        "generationConfig": {
            "temperature": 0.5,
            "maxOutputTokens": 300,
        },
    }

    try:
        started = time.monotonic()
        try:
            reply_text = extract_text(api_request(TEXT_MODEL, text_payload, timeout=15.0))
        except httpx.HTTPError as exc:
            print(f"Text model {TEXT_MODEL} failed: {exc}")
            reply_text = extract_text(api_request(TEXT_FALLBACK_MODEL, text_payload, timeout=25.0))
        print(f"Text reply completed in {time.monotonic() - started:.2f}s")
        if not reply_text:
            raise ValueError("Gemini returned no text reply.")

        current_history.append({"role": "model", "parts": [{"text": reply_text}]})
        current_history = current_history[-8:]

        tts_payload = {
            "contents": [{"role": "user", "parts": [{
                "text": reply_text,
                "speech_metadata": {
                    "style": "Speak naturally and warmly like a friendly Indian customer care executive. Use a clear, conversational phone-call tone."
                },
            }]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": "Aoede"}}},
            },
        }
        tts_data = api_request(TTS_MODEL, tts_payload)
        audio_bytes, mime_type = extract_audio(tts_data)
        if not audio_bytes:
            raise ValueError("Gemini returned no audio reply.")

        transcript = chat_ui or []
        transcript.append({"role": "user", "content": "[Voice message]"})
        transcript.append({"role": "assistant", "content": reply_text})
        return save_wav(audio_bytes, mime_type), transcript, current_history

    except (httpx.HTTPError, ValueError, KeyError) as exc:
        print(f"Solnce voice processing error: {exc}")
        transcript = chat_ui or []
        transcript.append({"role": "user", "content": "[Voice message]"})
        transcript.append({"role": "assistant", "content": "Sorry, I could not process that call. Please try again."})
        return None, transcript, current_history


def clear_call():
    return None, [], []


with gr.Blocks(title="Solnce Solar AI") as app:
    gr.Markdown("## ☀️ Solnce Solar — AI Phone Receptionist (Gujarati / Hindi / English)")
    gr.Markdown("Click the microphone to speak, then click **Stop**.")

    session_history = gr.State([])
    chatbot = gr.Chatbot(type="messages", height=300, label="Live Call Transcript")

    with gr.Row():
        mic = gr.Audio(sources=["microphone"], type="filepath", label="1. Speak Here")
        speaker = gr.Audio(label="2. Solnce Voice Response", autoplay=True)

    clear_btn = gr.Button("🔄 End Call / Start Fresh Conversation")

    mic.stop_recording(
        fn=process_voice,
        inputs=[mic, chatbot, session_history],
        outputs=[speaker, chatbot, session_history],
    )
    clear_btn.click(fn=clear_call, inputs=[], outputs=[speaker, chatbot, session_history])

if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860)),
        auth=(APP_USER, APP_PASS),
    )