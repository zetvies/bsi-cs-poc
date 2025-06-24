#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import asyncio
import os
import datetime
import io
import sys
import wave
import json
from contextlib import asynccontextmanager
from typing import Dict

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, WebSocket
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger
from pipecat_ai_small_webrtc_prebuilt.frontend import SmallWebRTCPrebuiltUI
from supabase import create_client, Client

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import TranscriptionMessage, TranscriptionUpdateFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor
from pipecat.processors.transcript_processor import TranscriptProcessor
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.transports.network.small_webrtc import SmallWebRTCTransport
from pipecat.transports.network.webrtc_connection import IceServer, SmallWebRTCConnection
from pipecat.transcriptions.language import Language

import aiofiles

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

# Debug environment variables
logger.info(f"🔍 Environment check:")
logger.info(f"   SUPABASE_URL: {'✅ Set' if supabase_url else '❌ Not set'}")
logger.info(f"   SUPABASE_KEY: {'✅ Set' if supabase_key else '❌ Not set'}")
logger.info(f"   OPENAI_API_KEY: {'✅ Set' if os.getenv('OPENAI_API_KEY') else '❌ Not set'}")
logger.info(f"   ELEVEN_API_KEY: {'✅ Set' if os.getenv('ELEVEN_API_KEY') else '❌ Not set'}")

supabase: Client = None

if supabase_url and supabase_key:
    supabase = create_client(supabase_url, supabase_key)
    logger.info("✅ Supabase client initialized")
else:
    logger.warning("⚠️ Supabase credentials not found. Transcriptions will not be saved to database.")

# Store connections by pc_id
pcs_map: Dict[str, SmallWebRTCConnection] = {}

ice_servers = [
    IceServer(
        urls="stun:stun.l.google.com:19302",
    )
]

# Mount the frontend at /
app.mount("/client", SmallWebRTCPrebuiltUI)


class SupabaseTranscriptHandler:
    """Handles saving transcriptions to Supabase database."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages = []
        self.conversation_started = False
        
    async def save_complete_conversation(self):
        """Save the complete conversation as a JSON array to Supabase."""
        if not supabase:
            logger.warning("Supabase not configured, skipping complete conversation save")
            return
            
        if not self.messages:
            logger.info("No messages to save")
            return
            
        try:
            # Convert TranscriptionMessage objects to dictionaries
            messages_dict = []
            for msg in self.messages:
                messages_dict.append({
                    "role": msg.role,
                    "content": msg.content
                })
            
            # Format messages as JSON array
            conversation_json = json.dumps(messages_dict, ensure_ascii=False, indent=2)
            
            # Prepare data for Supabase
            transcription_data = {
                "transcription_json": conversation_json
            }
            
            # Insert into iykra_transcriptions table
            result = supabase.table("iykra_transcriptions").insert(transcription_data).execute()
            logger.info(f"✅ Complete conversation saved to Supabase: {len(self.messages)} messages")
            logger.debug(f"✅ Conversation JSON: {conversation_json[:200]}...")
            
        except Exception as e:
            logger.error(f"❌ Error saving complete conversation to Supabase: {e}")
    
    async def on_transcript_update(self, processor: TranscriptProcessor, frame: TranscriptionUpdateFrame):
        """Handle new transcript messages."""
        logger.debug(f"Received transcript update with {len(frame.messages)} new messages")
        
        for msg in frame.messages:
            self.messages.append(msg)


async def save_audio(server_name: str, audio: bytes, sample_rate: int, num_channels: int):
    if len(audio) > 0:
        filename = (
            f"{server_name}_recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        )
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setsampwidth(2)
                wf.setnchannels(num_channels)
                wf.setframerate(sample_rate)
                wf.writeframes(audio)
            async with aiofiles.open(filename, "wb") as file:
                await file.write(buffer.getvalue())
        logger.info(f"Merged audio saved to {filename}")
    else:
        logger.info("No audio data to save")


async def run_bot(webrtc_connection: SmallWebRTCConnection, mode="inbound"):
    logger.info(f"Starting BSI Syariah bot (CLEAN VERSION) with mode: {mode}")

    # Create a transport using the WebRTC connection - EXACTLY like examples
    transport = SmallWebRTCTransport(
        webrtc_connection=webrtc_connection,
        params=TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            # Enhanced audio processing to reduce static noise
            echo_cancellation=True,
            noise_suppression=True,
            automatic_gain_control=True,
            # Additional noise reduction settings
            high_pass_filter=True,  # Remove low-frequency noise
            # Audio quality settings
            audio_in_sample_rate=16000,  # Higher sample rate for better quality
            audio_out_sample_rate=16000,
            audio_in_channels=1,  # Mono for better processing
            audio_out_channels=1,
            # WebRTC specific audio constraints
            audio_constraints={
                "echoCancellation": True,
                "noiseSuppression": True,
                "autoGainControl": True,
                "highPassFilter": True,
                "sampleRate": 16000,
                "channelCount": 1,
            }
        ),
    )

    # Use EXACTLY the same STT configuration as the working Twilio version
    stt = OpenAISTTService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-transcribe",  # Using the latest GPT-4o transcription model
        language=Language.ID,  # Indonesian language
        temperature=0.0,  # For more accurate transcription
        prompt="""Transcribe accurately in Bahasa Indonesia, especially for banking and financial terms. 

Important terms to transcribe correctly:
- BSI Syariah (not Bank Sentral)
- Bank Syariah Indonesia
- Tabungan BSI
- Deposito Syariah
- KPR Syariah
- Kartu Kredit Syariah
- Cashback
- Suku bunga
- DP (Down Payment)
- Biaya tahunan

Please transcribe BSI Syariah exactly as spoken, not as 'Bank Sentral'."""
    )
    
    # Force the STT service to be used
    logger.info(f"✅ Using STT service: {type(stt).__name__}")
    logger.info(f"✅ STT service ID: {id(stt)}")
    logger.info(f"✅ STT service config: {stt}")
    
    # Add event handlers to track STT usage
    @stt.event_handler("on_transcription")
    async def on_transcription(stt, transcription):
        logger.info(f"🎯 OUR STT Transcription: '{transcription.text}'")
        logger.info(f"🎯 OUR STT Confidence: {transcription.confidence}")
        logger.info(f"🎯 OUR STT Language: {transcription.language}")

    @stt.event_handler("on_transcription_start")
    async def on_transcription_start(stt):
        logger.info("🎯 OUR STT started transcribing")

    @stt.event_handler("on_transcription_end")
    async def on_transcription_end(stt):
        logger.info("🎯 OUR STT finished transcribing")

    @stt.event_handler("on_transcription_error")
    async def on_transcription_error(stt, error):
        logger.error(f"🎯 OUR STT Error: {error}")

    tts = ElevenLabsTTSService(
        api_key=os.getenv("ELEVEN_API_KEY"),
        voice_id=os.getenv("ELEVEN_VOICE_ID"),
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
    )

    # Audio buffer processor for recording
    audiobuffer = AudioBufferProcessor(user_continuous_stream=True)
    
    # Transcript processor for Supabase storage
    transcript = TranscriptProcessor()
    
    # Create Supabase transcript handler with session ID
    session_id = f"webrtc_{webrtc_connection.pc_id}"
    transcript_handler = SupabaseTranscriptHandler(session_id)

    # Full detailed system prompt (context, rules, promo)
    detailed_prompt = (
        "Anda adalah asisten penjualan BSI Syariah bernama Melina. Anda harus selalu berbicara dalam Bahasa Indonesia dengan sopan dan profesional. \n\n"
        "Tugas utama Anda adalah:\n"
        "1. Memperkenalkan produk dan layanan BSI Syariah\n"
        "2. Menjelaskan promo-promo yang sedang berlangsung\n"
        "3. Membantu calon nasabah memahami manfaat produk syariah\n"
        "4. Mengumpulkan informasi dasar untuk follow-up dari tim sales\n\n"
        "Promo yang sedang berlangsung:\n"
        "- Tabungan BSI Syariah: Buka rekening baru dapatkan cashback Rp 100.000\n"
        "- Deposito Syariah: Suku bunga 5% p.a. untuk tenor 3 bulan\n"
        "- KPR Syariah: DP 0% untuk KPR pertama\n"
        "- Kartu Kredit Syariah: Gratis biaya tahunan tahun pertama\n\n"
        "Panduan dalam menjawab:\n"
        "1. 'Assalamualaikum' dan perkenalan singkat. hanya katakan di awal pembicaraan. tidak perlu menjawab waalaikumsalam\n"
        "2. Tanyakan kebutuhan nasabah dengan sopan\n"
        "3. Jelaskan promo yang relevan dengan kebutuhan mereka\n"
        "4. Hindari karakter khusus dalam jawaban karena akan dikonversi ke audio\n"
        "5. Berikan jawaban yang singkat, jelas, dan informatif\n"
        "6. Jika nasabah tertarik, tanyakan nama dan nomor telepon untuk follow-up\n\n"
        "Ingat: Anda adalah perwakilan BSI Syariah, jadi selalu jaga etika dan profesionalisme dalam setiap percakapan."
    )

    # Set system prompt and forced first assistant message based on mode
    if mode == "inbound":
        opening = "Assalamualaikum, selamat datang di layanan customer service BSI Syariah. Perkenalkan saya Melina dari customer service BSI Syariah. Silakan sampaikan kebutuhan Anda."
        system_prompt = opening + "\n\n" + detailed_prompt
        first_assistant = opening
    elif mode == "outbound":
        opening = "Assalamualaikum, perkenalkan saya Melina dari customer service BSI Syariah. Boleh minta waktunya untuk menjelaskan produk BSI Syariah?"
        system_prompt = opening + "\n\n" + detailed_prompt
        first_assistant = opening
    elif mode == "free":
        opening = "Assalamualaikum, perkenalkan saya Melina dari customer service BSI Syariah. Silakan mulai percakapan dengan asisten BSI Syariah."
        system_prompt = opening + "\n\n" + detailed_prompt
        first_assistant = opening
    else:
        opening = "Assalamualaikum, perkenalkan saya Melina dari customer service BSI Syariah. Anda terhubung dengan layanan BSI Syariah."
        system_prompt = opening + "\n\n" + detailed_prompt
        first_assistant = opening

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "assistant", "content": first_assistant},
    ]

    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # Create the pipeline with explicit STT service - EXACTLY like examples
    pipeline = Pipeline(
        [
            transport.input(),  # WebRTC input from client
            stt,  # Speech-To-Text (EXPLICITLY our OpenAISTTService)
            transcript.user(),  # User transcripts for Supabase
            context_aggregator.user(),
            llm,  # LLM
            tts,  # Text-To-Speech
            transport.output(),  # WebRTC output to client
            audiobuffer,  # Used to buffer the audio in the pipeline (moved to end)
            transcript.assistant(),  # Assistant transcripts for Supabase
            context_aggregator.assistant(),
        ]
    )

    # Use EXACTLY the same task configuration as the working Twilio version
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
            # Enhanced audio quality parameters
            audio_in_sample_rate=16000,
            audio_out_sample_rate=16000,
            audio_in_channels=1,
            audio_out_channels=1,
            # Additional audio processing
            audio_in_frame_size=480,  # Smaller frame size for better processing
            audio_out_frame_size=480,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Start recording.
        await audiobuffer.start_recording()
        logger.info("✅ Recording started")
        logger.info("✅ Audio processing enabled: echo cancellation, noise suppression, auto gain control")
        # Kick off the conversation.
        messages.append({"role": "system", "content": "Please introduce yourself to the user."})
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Client disconnected")
        # Save complete conversation to Supabase
        await transcript_handler.save_complete_conversation()
        await task.cancel()
    
    # Add audio debugging handlers
    @transport.event_handler("on_audio_in")
    async def on_audio_in(transport, audio_data):
        if audio_data:
            audio_level = max(abs(sample) for sample in audio_data) if len(audio_data) > 0 else 0
            logger.debug(f"🎤 Audio input level: {audio_level:.4f} (length: {len(audio_data)})")

    @transport.event_handler("on_audio_out")
    async def on_audio_out(transport, audio_data):
        if audio_data:
            audio_level = max(abs(sample) for sample in audio_data) if len(audio_data) > 0 else 0
            logger.debug(f"🔊 Audio output level: {audio_level:.4f} (length: {len(audio_data)})")

    @audiobuffer.event_handler("on_audio_data")
    async def on_audio_data(buffer, audio, sample_rate, num_channels):
        server_name = f"webrtc_server_{webrtc_connection.pc_id}"
        await save_audio(server_name, audio, sample_rate, num_channels)
        logger.debug(f"💾 Recording audio: {len(audio)} bytes, {sample_rate}Hz, {num_channels} channels")
    
    # Register transcript update handler for Supabase
    @transcript.event_handler("on_transcript_update")
    async def on_transcript_update(processor, frame):
        await transcript_handler.on_transcript_update(processor, frame)

    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


@app.get("/", include_in_schema=False)
async def root_redirect():
    return RedirectResponse(url="/static/dashboard.html")


@app.get("/webrtc", include_in_schema=False)
async def webrtc_redirect():
    return RedirectResponse(url="/static/dashboard.html")


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    pc_id = request.get("pc_id")
    mode = request.get("mode", "inbound")  # default to inbound if not set

    if pc_id and pc_id in pcs_map:
        pipecat_connection = pcs_map[pc_id]
        logger.info(f"Reusing existing connection for pc_id: {pc_id}")
        await pipecat_connection.renegotiate(
            sdp=request["sdp"],
            type=request["type"],
            restart_pc=request.get("restart_pc", False),
        )
    else:
        pipecat_connection = SmallWebRTCConnection(ice_servers)
        await pipecat_connection.initialize(sdp=request["sdp"], type=request["type"])

        @pipecat_connection.event_handler("closed")
        async def handle_disconnected(webrtc_connection: SmallWebRTCConnection):
            logger.info(f"Discarding peer connection for pc_id: {webrtc_connection.pc_id}")
            pcs_map.pop(webrtc_connection.pc_id, None)

        # Pass mode to run_bot
        background_tasks.add_task(run_bot, pipecat_connection, mode)

    answer = pipecat_connection.get_answer()
    pcs_map[answer["pc_id"]] = pipecat_connection
    return answer


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # Run app
    coros = [pc.disconnect() for pc in pcs_map.values()]
    await asyncio.gather(*coros)
    pcs_map.clear()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="BSI Syariah WebRTC Bot")
    parser.add_argument(
        "--host", default="localhost", help="Host for HTTP server (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port for HTTP server (default: 7860)"
    )
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port) 