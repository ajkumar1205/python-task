import os
import json
import asyncio
import websockets
import pyaudio
import aiohttp
from dotenv import load_dotenv
from typing import AsyncIterator

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEETS_API_KEY = os.getenv("NEETSAI_API_KEY")

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 8000

audio_queue = asyncio.Queue()

async def play_stream(audio_stream: AsyncIterator[bytes]) -> bytes:
    print("Playing audio...")
    process = await asyncio.create_subprocess_exec(
        "mpv", "--no-cache", "--no-terminal", "--", "fd://0",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )

    audio = b""

    async for chunk in audio_stream:
        if chunk is not None:
            process.stdin.write(chunk)
            await process.stdin.drain()
            audio += chunk

    if process.stdin:
        process.stdin.close()

    await process.wait()

async def say(text: str):
    print("Requesting TTS...")
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.neets.ai/v1/tts",
            headers={
                "Content-Type": "application/json",
                "X-API-Key": NEETS_API_KEY
            },
            json={
                "text": text,
                "voice_id": "us-female-2",
                "params": {
                    "model": "style-diff-500"
                }
            }
        ) as response:
            audio_stream = response.content.iter_any()
            print("Received audio stream....")
            await play_stream(audio_stream)

async def get_llm_response(prompt):
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "mixtral-8x7b-32768",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.7,
                "max_tokens": 150
            }
        ) as response:
            result = await response.json()
            return result['choices'][0]['message']['content']

def mic_callback(input_data, frame_count, time_info, status_flag):
    audio_queue.put_nowait(input_data)
    return (input_data, pyaudio.paContinue)

async def run_transcription():
    deepgram_url = f'wss://api.deepgram.com/v1/listen?punctuate=true&encoding=linear16&sample_rate=16000'

    async with websockets.connect(
        deepgram_url, extra_headers={"Authorization": f"Token {DEEPGRAM_API_KEY}"}
    ) as ws:
        print("Connected to Deepgram")

        async def sender(ws):
            print("Ready to stream audio. Speak into your microphone.")
            try:
                while True:
                    if not audio_queue.empty():
                        mic_data = await audio_queue.get()
                        await ws.send(mic_data)
                    else:
                        await asyncio.sleep(0.01)
            except websockets.exceptions.ConnectionClosedOK:
                print("Closed Deepgram connection")
            except Exception as e:
                print(f"Error while sending: {str(e)}")
                raise

        async def receiver(ws):
            transcript = ""
            try:
                async for msg in ws:
                    res = json.loads(msg)
                    if res.get("is_final"):
                        transcript = res.get("channel", {}).get("alternatives", [{}])[0].get("transcript", "")
                        if transcript:
                            print(f"Transcription: {transcript}")
            except websockets.exceptions.ConnectionClosed:
                print("Deepgram connection closed")
            return transcript

        audio = pyaudio.PyAudio()
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            stream_callback=mic_callback,
        )
        stream.start_stream()

        sender_task = asyncio.create_task(sender(ws))
        receiver_task = asyncio.create_task(receiver(ws))

        # Record for 7 seconds
        await asyncio.sleep(7)
        stream.stop_stream()
        stream.close()
        audio.terminate()

        await ws.close()
        await asyncio.gather(sender_task, receiver_task)

        return receiver_task.result()

async def main():
    try:
        transcript = await run_transcription()
        if transcript:
            print(f"Full transcription: {transcript}")
            
            llm_response = await get_llm_response(transcript)
            print(f"LLM Response: {llm_response}")
            
            await say(llm_response)
        else:
            print("No transcription received.")
        
        print("Process completed.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())