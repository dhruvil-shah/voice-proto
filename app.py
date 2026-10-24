import os
import wave
import tempfile
import gradio as gr
from google import genai
from google.genai import types

API_KEY = os.environ.get("GEMINI_API_KEY")
APP_USER = os.environ.get("APP_USER")
APP_PASS = os.environ.get("APP_PASS")

if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set.")
if not APP_USER or not APP_PASS:
    raise RuntimeError("APP_USER and APP_PASS environment variables must be set.")

client = genai.Client(api_key=API_KEY)

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


def save_pcm_to_wav(pcm_bytes, rate=24000):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(pcm_bytes)
    return tmp.name


def process_voice(audio_filepath, chat_ui, user_history):
    if not audio_filepath:
        return None, chat_ui, user_history

    with open(audio_filepath, "rb") as f:
        wav_bytes = f.read()

    user_history.append(
        types.Content(role="user", parts=[types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav")])
    )

    text_resp = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=user_history,
        config=types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.5),
    )
    reply_text = text_resp.text.strip()

    user_history.append(
        types.Content(role="model", parts=[types.Part.from_text(text=reply_text)])
    )

    audio_resp = client.models.generate_content(
        model="gemini-2.5-flash-preview-tts",
        contents=f"Speak this naturally and warmly like a friendly Indian customer care executive on a phone call: {reply_text}",
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Aoede")
                )
            ),
        ),
    )

    pcm_data = audio_resp.candidates[0].content.parts[0].inline_data.data
    out_wav_path = save_pcm_to_wav(pcm_data)

    chat_ui = chat_ui or []
    chat_ui.append({"role": "user", "content": "🎤 [Voice Message]"})
    chat_ui.append({"role": "assistant", "content": reply_text})

    return out_wav_path, chat_ui, user_history


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