#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import datetime
import io
import json
import os
import sys
import wave
import re
from contextlib import asynccontextmanager
from typing import Dict

import aiofiles
import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, WebSocket
from fastapi.responses import RedirectResponse, FileResponse
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

import sqlite3
from pathlib import Path

# Try to import openai, but don't fail if not available
try:
    import openai
except ImportError:
    openai = None

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize SQLite database
def init_database():
    """Initialize SQLite database and create tables if they don't exist"""
    db_path = Path("transcriptions.db")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create transcriptions table (no audio_file column)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transcriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            transcription_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    # Remove audio_file column if it exists (SQLite doesn't support DROP COLUMN directly, so skip for now)
    # If you want to fully remove it, you need to recreate the table and copy data.
    
    conn.commit()
    conn.close()
    logger.info(f"✅ SQLite database initialized at {db_path}")

# Initialize database on startup
init_database()

# Store connections by pc_id
pcs_map: Dict[str, SmallWebRTCConnection] = {}

ice_servers = [
    IceServer(
        urls="stun:stun.l.google.com:19302",
    )
]

# Mount the frontend at /
app.mount("/client", SmallWebRTCPrebuiltUI)

class SQLiteTranscriptHandler:
    """Handles saving transcriptions to SQLite database."""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages = []
        self.conversation_started = False
        self.audio_file = None
        
    def set_audio_file(self, audio_file: str):
        """Set the audio file name for this session."""
        self.audio_file = audio_file
        logger.info(f"✅ Updated audio filename for session {self.session_id}: {audio_file}")
        
    async def save_complete_conversation(self):
        """Save the complete conversation as a JSON array to SQLite with processed data."""
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
            
            # Always generate an audio filename when saving (even if no audio was recorded)
            if not self.audio_file:
                self.audio_file = f"recording_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.wav"
                logger.info(f"✅ Generated audio filename for session {self.session_id}: {self.audio_file}")
            
            logger.info(f"🔍 About to process conversation data with audio_file: {self.audio_file}")
            
            # Process the conversation data (including audio_file)
            processed_data = await self.process_conversation_data(messages_dict, self.audio_file)
            logger.info(f"✅ Adding audio file to transcription: {self.audio_file}")
            logger.info(f"🔍 Processed data keys: {list(processed_data.keys())}")
            logger.info(f"🔍 Audio file in processed_data: {processed_data.get('audio_file')}")
            
            # Save to SQLite
            conn = sqlite3.connect("transcriptions.db")
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO transcriptions (session_id, transcription_json)
                VALUES (?, ?)
            ''', (self.session_id, json.dumps(processed_data, ensure_ascii=False, indent=2)))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Complete conversation saved to SQLite: {len(self.messages)} messages with processed data")
            logger.info(f"✅ Audio file saved to JSON: {self.audio_file}")
            logger.debug(f"✅ Processed data: {json.dumps(processed_data, ensure_ascii=False)[:200]}...")
            
        except Exception as e:
            logger.error(f"❌ Error saving complete conversation to SQLite: {e}")
    
    async def process_conversation_data(self, messages, audio_file=None):
        """Process conversation data to extract sentiment, summary, and other analytics."""
        try:
            # Get user messages for analysis
            user_messages = [msg for msg in messages if msg['role'] == 'user']
            user_text = '\n'.join([msg['content'] for msg in user_messages])
            
            # Check if we have enough user content to analyze
            if not user_text.strip():
                logger.warning("No user content to analyze, returning basic data")
                return {
                    "messages": messages,
                    "message_count": len(messages),
                    "user_message_count": len(user_messages),
                    "audio_file": audio_file
                }
            
            # Analyze sentiment
            sentiment = await self.analyze_sentiment(user_text)
            
            # Generate summary
            summary = await self.generate_summary(user_text)
            
            # Check if analysis was successful
            if not self.is_analysis_successful(sentiment, summary):
                logger.warning("Analysis failed or returned default values, returning basic data")
                return {
                    "messages": messages,
                    "message_count": len(messages),
                    "user_message_count": len(user_messages),
                    "audio_file": audio_file
                }
            
            # Calculate other metrics
            disposition = self.determine_disposition(messages)
            compliance = self.check_sharia_compliance(messages)
            lead_intent = self.detect_lead_intent(messages)
            total_score = self.calculate_total_score(sentiment['score'], compliance, lead_intent)
            
            return {
                "messages": messages,
                "sentiment": sentiment,
                "summary": summary,
                "disposition": disposition,
                "compliance": compliance,
                "lead_intent": lead_intent,
                "total_score": total_score,
                "message_count": len(messages),
                "user_message_count": len(user_messages),
                "audio_file": audio_file
            }
            
        except Exception as e:
            logger.error(f"Error processing conversation data: {e}")
            # Return basic data if processing fails
            return {
                "messages": messages,
                "message_count": len(messages),
                "user_message_count": len([msg for msg in messages if msg['role'] == 'user']),
                "audio_file": audio_file
            }
    
    def is_analysis_successful(self, sentiment, summary):
        """Check if the analysis was actually successful and not just default values."""
        # Check if sentiment analysis was successful
        sentiment_successful = (
            sentiment and 
            isinstance(sentiment, dict) and
            sentiment.get('sentiment') in ['positive', 'negative', 'neutral'] and
            isinstance(sentiment.get('score'), (int, float)) and
            0 <= sentiment.get('score', 0) <= 100
        )
        
        # Check if summary was successful
        summary_successful = (
            summary and 
            isinstance(summary, dict) and
            summary.get('interest_summary') and 
            summary.get('interest_summary') != "Minat pelanggan perlu dianalisis lebih lanjut" and
            summary.get('followup_recommendation') and
            summary.get('followup_recommendation') != "Hubungi kembali pelanggan untuk informasi lebih detail"
        )
        
        return sentiment_successful and summary_successful
    
    async def analyze_sentiment(self, user_text):
        """Analyze sentiment using OpenAI."""
        try:
            if not openai:
                logger.warning("OpenAI not available, using default sentiment")
                return {"sentiment": "neutral", "score": 50}
                
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "Analyze the sentiment of this conversation and return a JSON object with a 'sentiment' field (positive, neutral, or negative) and a 'score' field (0-100)."
                    },
                    {
                        "role": "user",
                        "content": user_text
                    }
                ]
            )
            
            analysis = json.loads(response.choices[0].message.content)
            return {
                "sentiment": analysis.get("sentiment", "neutral"),
                "score": analysis.get("score", 50)
            }
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {"sentiment": "neutral", "score": 50}
    
    async def generate_summary(self, user_text):
        """Generate summary using OpenAI."""
        try:
            if not openai:
                logger.warning("OpenAI not available, using default summary")
                return {
                    "interest_summary": "Minat pelanggan perlu dianalisis lebih lanjut",
                    "followup_recommendation": "Hubungi kembali pelanggan untuk informasi lebih detail"
                }
                
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            
            prompt = f"""
You are a customer service assistant. Given the following conversation, summarize the customer's interest in 1-2 sentences, and give a specific, actionable follow-up recommendation in Bahasa Indonesia. 
Selalu kembalikan JSON object dengan dua field:
- interest_summary (ringkasan minat pelanggan)
- followup_recommendation (tindakan lanjutan yang spesifik, misal: 'Hubungi kembali dengan detail program KPR').

Contoh output:
{{
  "interest_summary": "Pelanggan tertarik dengan produk KPR BSI Syariah.",
  "followup_recommendation": "Hubungi kembali pelanggan dan berikan detail program KPR."
}}

Conversation:
{user_text}
"""
            
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": prompt
                    }
                ]
            )
            
            summary = json.loads(response.choices[0].message.content)
            return {
                "interest_summary": summary.get("interest_summary", "Minat pelanggan perlu dianalisis lebih lanjut"),
                "followup_recommendation": summary.get("followup_recommendation", "Hubungi kembali pelanggan untuk informasi lebih detail")
            }
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return {
                "interest_summary": "Minat pelanggan perlu dianalisis lebih lanjut",
                "followup_recommendation": "Hubungi kembali pelanggan untuk informasi lebih detail"
            }
    
    def determine_disposition(self, messages):
        """Determine call disposition based on messages."""
        has_phone_number = any(
            msg['role'] == 'user' and re.search(r'\d{10,}', msg['content'])
            for msg in messages
        )
        has_interest = any(
            msg['role'] == 'user' and 
            ('tertarik' in msg['content'].lower() or 'ya' in msg['content'].lower())
            for msg in messages
        )
        
        if has_phone_number and has_interest:
            return 'success'
        elif has_interest:
            return 'followup'
        else:
            return 'rejected'
    
    def check_sharia_compliance(self, messages):
        """Check for sharia compliance keywords."""
        keywords = [r'riba', r'gharar', r'maysir', r'judi', r'bunga', r'haram']
        all_text = ' '.join([msg['content'] for msg in messages])
        
        for keyword in keywords:
            if re.search(keyword, all_text, re.IGNORECASE):
                return 'Flag'
        return 'Pass'
    
    def detect_lead_intent(self, messages):
        """Detect lead intent from conversation."""
        triggers = [
            {'tag': 'KPR', 'regex': r'kpr|kredit pemilikan rumah|mortgage'},
            {'tag': 'Tabungan', 'regex': r'tabungan|rekening baru|open account'},
            {'tag': 'Deposito', 'regex': r'deposito'},
            {'tag': 'Kartu Kredit', 'regex': r'kartu kredit|credit card'}
        ]
        
        all_text = ' '.join([msg['content'] for msg in messages])
        for trigger in triggers:
            if re.search(trigger['regex'], all_text, re.IGNORECASE):
                return trigger['tag']
        return 'None'
    
    def calculate_total_score(self, sentiment_score, compliance, lead_intent):
        """Calculate total conversation score."""
        s = sentiment_score / 100
        c = 1 if compliance == 'Pass' else 0
        l = 1 if lead_intent != 'None' else 0
        return round(s * 40 + c * 30 + l * 30)
    
    async def on_transcript_update(self, processor: TranscriptProcessor, frame: TranscriptionUpdateFrame):
        """Handle new transcript messages."""
        logger.debug(f"Received transcript update with {len(frame.messages)} new messages")
        
        for msg in frame.messages:
            self.messages.append(msg)


async def save_audio(server_name: str, audio: bytes, sample_rate: int, num_channels: int):
    logger.info(f"🎤 save_audio called: {len(audio)} bytes, {sample_rate}Hz, {num_channels} channels")
    
    # Always generate filename in the required format
    filename = f"recording_{datetime.datetime.now().strftime('%Y%m%d%H%M')}.wav"
    logger.info(f"🎤 Generated audio filename: {filename}")
    
    if len(audio) > 0:
        logger.info(f"🎤 Creating audio file: {filename}")
        with io.BytesIO() as buffer:
            with wave.open(buffer, "wb") as wf:
                wf.setsampwidth(2)
                wf.setnchannels(num_channels)
                wf.setframerate(sample_rate)
                wf.writeframes(audio)
            async with aiofiles.open(filename, "wb") as file:
                await file.write(buffer.getvalue())
        logger.info(f"✅ Merged audio saved to {filename}")
    else:
        logger.warning("⚠️ No audio data to save - audio buffer is empty, but filename generated")
    
    # Always return the filename
    return filename


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

    # STT Service with retry logic
    stt = OpenAISTTService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="whisper-1",  # Using stable whisper-1 instead of gpt-4o-transcribe
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

Please transcribe BSI Syariah exactly as spoken, not as 'Bank Sentral'.""",
        # Add retry configuration
        max_retries=3,
        retry_delay=1.0,
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
    
    # Add TTS debugging
    logger.info(f"✅ Using TTS service: {type(tts).__name__}")
    logger.info(f"✅ TTS service ID: {id(tts)}")
    logger.info(f"✅ TTS API Key: {'✅ Set' if os.getenv('ELEVEN_API_KEY') else '❌ Not set'}")
    logger.info(f"✅ TTS Voice ID: {'✅ Set' if os.getenv('ELEVEN_VOICE_ID') else '❌ Not set'}")
    
    @tts.event_handler("on_tts_start")
    async def on_tts_start(tts, text):
        logger.info(f"🔊 TTS Started: '{text}'")
    
    @tts.event_handler("on_tts_end")
    async def on_tts_end(tts, audio):
        logger.info(f"🔊 TTS Ended: {len(audio) if audio else 0} bytes of audio")
    
    @tts.event_handler("on_tts_error")
    async def on_tts_error(tts, error):
        logger.error(f"🔊 TTS Error: {error}")

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
    )
    
    # Add LLM debugging
    logger.info(f"✅ Using LLM service: {type(llm).__name__}")
    logger.info(f"✅ LLM service ID: {id(llm)}")
    logger.info(f"✅ LLM model: gpt-4o-mini")
    
    @llm.event_handler("on_llm_response")
    async def on_llm_response(llm, response):
        logger.info(f"🤖 LLM Response: '{response.content}'")
    
    @llm.event_handler("on_llm_error")
    async def on_llm_error(llm, error):
        logger.error(f"🤖 LLM Error: {error}")

    # Audio buffer processor for recording
    audiobuffer = AudioBufferProcessor(user_continuous_stream=True)
    
    # Transcript processor for Supabase storage
    transcript = TranscriptProcessor()
    
    # Create SQLite transcript handler with session ID
    session_id = f"webrtc_{webrtc_connection.pc_id}"
    transcript_handler = SQLiteTranscriptHandler(session_id)

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
        "5. Berikan jawaban yang singkat, efisien, jelas, dan informatif\n"
        "6. Jangan memanggil nama nasabah. Jika anda bisa membedakan jenis kelamin, maka anda bisa memanggilnya dengan 'Pak' atau 'Bu' sesuai dengan jenis kelamin nasabah. Jika tidak bisa membedakan, maka anda bisa memanggilnya dengan 'Mas' atau 'Mba'.\n"
        "Ingat: Anda adalah perwakilan BSI Syariah, jadi selalu jaga etika dan profesionalisme dalam setiap percakapan."
    )

    # Set system prompt and forced first assistant message based on mode
    if mode == "inbound":
        opening = "Assalamualaikum, selamat datang di layanan customer service BSI Syariah. Perkenalkan saya Melina dari customer service BSI Syariah. Ada yang bisa kami bantu?"
        system_prompt = opening + "\n\n" + detailed_prompt
        first_assistant = opening
    elif mode == "outbound":
        opening = "Assalamualaikum, perkenalkan saya Melina dari customer service BSI Syariah. Boleh minta waktunya untuk menjelaskan produk BSI Syariah?"
        system_prompt = opening + "\n\n" + detailed_prompt
        first_assistant = opening
    elif mode == "free":
        opening = "Assalamualaikum, perkenalkan saya Melina dari customer service BSI Syariah. Ada yang bisa kami bantu?."
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
        # Save complete conversation to SQLite
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
        filename = await save_audio(server_name, audio, sample_rate, num_channels)
        transcript_handler.set_audio_file(filename)
        logger.info(f"✅ Audio file set for session {session_id}: {filename}")
        logger.debug(f"🎤 Recording audio: {len(audio)} bytes, {sample_rate}Hz, {num_channels} channels")
    
    # Register transcript update handler for SQLite
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


@app.get("/api/transcriptions")
async def get_transcriptions():
    """Get all transcriptions from SQLite database"""
    try:
        conn = sqlite3.connect("transcriptions.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, session_id, transcription_json, created_at 
            FROM transcriptions 
            ORDER BY created_at DESC
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        transcriptions = []
        for row in rows:
            transcription_id, session_id, transcription_json, created_at = row
            try:
                messages = json.loads(transcription_json)
                audio_file = messages.get("audio_file")
                transcriptions.append({
                    "id": transcription_id,
                    "session_id": session_id,
                    "messages": messages,
                    "audio_file": audio_file,
                    "created_at": created_at,
                    "message_count": len(messages.get('messages', []))
                })
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON for transcription {transcription_id}")
                continue
        
        return {"transcriptions": transcriptions}
        
    except Exception as e:
        logger.error(f"Error fetching transcriptions: {e}")
        return {"error": "Failed to fetch transcriptions"}

@app.get("/api/transcriptions/{transcription_id}")
async def get_transcription(transcription_id: int):
    """Get a specific transcription by ID"""
    try:
        conn = sqlite3.connect("transcriptions.db")
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT session_id, transcription_json, created_at 
            FROM transcriptions 
            WHERE id = ?
        ''', (transcription_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            session_id, transcription_json, created_at = row
            messages = json.loads(transcription_json)
            audio_file = messages.get("audio_file")
            return {
                "id": transcription_id,
                "session_id": session_id,
                "messages": messages,
                "audio_file": audio_file,
                "created_at": created_at
            }
        else:
            return {"error": "Transcription not found"}
            
    except Exception as e:
        logger.error(f"Error fetching transcription {transcription_id}: {e}")
        return {"error": "Failed to fetch transcription"}

@app.get("/audio/{filename}")
async def get_audio_file(filename: str):
    """Serve audio files"""
    try:
        file_path = Path(filename)
        if file_path.exists() and file_path.suffix.lower() == '.wav':
            return FileResponse(
                path=file_path,
                media_type='audio/wav',
                filename=filename
            )
        else:
            return {"error": "Audio file not found"}
    except Exception as e:
        logger.error(f"Error serving audio file {filename}: {e}")
        return {"error": "Failed to serve audio file"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BSI Syariah WebRTC Bot")
    parser.add_argument(
        "--host", default="localhost", help="Host for HTTP server (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port for HTTP server (default: 7860)"
    )
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port) 