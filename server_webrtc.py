#!/usr/bin/env python3
#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import os
import json
import sqlite3
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Dict

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import RedirectResponse, HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from bot_webrtc_clean import run_bot
from pipecat.transports.network.webrtc_connection import IceServer, SmallWebRTCConnection

load_dotenv(override=True)

app = FastAPI()

# Initialize SQLite database
def init_database():
    """Initialize SQLite database and create tables if they don't exist"""
    db_path = Path("transcriptions.db")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create transcriptions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS transcriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            transcription_json TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
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

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
async def startup_event():
    """Set default values on startup"""
    if not hasattr(app.state, 'testing'):
        app.state.testing = False


@app.get("/", include_in_schema=False)
async def root_redirect():
    """Serve the summary dashboard by default"""
    from pathlib import Path
    summary_file = Path(__file__).parent / "static" / "dashboard.html"
    if summary_file.exists():
        return HTMLResponse(content=summary_file.read_text(), media_type="text/html")
    else:
        return HTMLResponse(content="""
        <html>
        <head><title>BSI Syariah Dashboard</title></head>
        <body>
            <h1>BSI Syariah Dashboard</h1>
            <p>Summary dashboard not found.</p>
            <p><a href="/call">Go to Voice Chat</a></p>
        </body>
        </html>
        """)


@app.get("/call", include_in_schema=False)
async def call_page():
    """Serve the call page"""
    from pathlib import Path
    call_file = Path(__file__).parent / "static" / "call.html"
    if call_file.exists():
        return HTMLResponse(content=call_file.read_text(), media_type="text/html")
    else:
        return HTMLResponse(content="""
        <html>
        <head><title>BSI Syariah Call</title></head>
        <body>
            <h1>BSI Syariah Call</h1>
            <p>Call page not found.</p>
            <p><a href="/">Go to Dashboard</a></p>
        </body>
        </html>
        """)


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    pc_id = request.get("pc_id")
    mode = request.get("mode", "inbound")  # Get mode from request, default to "inbound"

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

        # Run bot function with proper mode string
        background_tasks.add_task(run_bot, pipecat_connection, mode)

    answer = pipecat_connection.get_answer()
    # Updating the peer connection inside the map
    pcs_map[answer["pc_id"]] = pipecat_connection

    return answer


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
                processed_data = json.loads(transcription_json)
                
                # Handle both old format (just messages) and new format (processed data)
                if isinstance(processed_data, list):
                    # Old format - just messages array
                    messages = processed_data
                    transcriptions.append({
                        "id": transcription_id,
                        "session_id": session_id,
                        "messages": messages,
                        "created_at": created_at,
                        "message_count": len(messages)
                    })
                else:
                    # New format - processed data (may or may not have analytics)
                    transcriptions.append({
                        "id": transcription_id,
                        "session_id": session_id,
                        "messages": processed_data.get("messages", []),
                        "created_at": created_at,
                        "message_count": processed_data.get("message_count", 0),
                        "sentiment": processed_data.get("sentiment"),
                        "summary": processed_data.get("summary"),
                        "disposition": processed_data.get("disposition"),
                        "compliance": processed_data.get("compliance"),
                        "lead_intent": processed_data.get("lead_intent"),
                        "total_score": processed_data.get("total_score"),
                        "audio_file": processed_data.get("audio_file")
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
            processed_data = json.loads(transcription_json)
            
            # Handle both old format (just messages) and new format (processed data)
            if isinstance(processed_data, list):
                # Old format - just messages array
                messages = processed_data
                return {
                    "id": transcription_id,
                    "session_id": session_id,
                    "messages": messages,
                    "created_at": created_at
                }
            else:
                # New format - processed data
                return {
                    "id": transcription_id,
                    "session_id": session_id,
                    "messages": processed_data,
                    "audio_file": processed_data.get("audio_file"),
                    "created_at": created_at
                }
        else:
            return {"error": "Transcription not found"}
            
    except Exception as e:
        logger.error(f"Error fetching transcription {transcription_id}: {e}")
        return {"error": "Failed to fetch transcription"}


@app.delete("/api/transcriptions/{transcription_id}")
async def delete_transcription(transcription_id: int):
    """Delete a specific transcription by ID"""
    try:
        conn = sqlite3.connect("transcriptions.db")
        cursor = conn.cursor()
        
        # Check if transcription exists
        cursor.execute('SELECT id FROM transcriptions WHERE id = ?', (transcription_id,))
        if not cursor.fetchone():
            conn.close()
            return {"error": "Transcription not found"}
        
        # Delete the transcription
        cursor.execute('DELETE FROM transcriptions WHERE id = ?', (transcription_id,))
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Transcription {transcription_id} deleted successfully")
        return {"success": True, "message": f"Transcription {transcription_id} deleted successfully"}
        
    except Exception as e:
        logger.error(f"Error deleting transcription {transcription_id}: {e}")
        return {"error": "Failed to delete transcription"}


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield  # Run app
    coros = [pc.disconnect() for pc in pcs_map.values()]
    await asyncio.gather(*coros)
    pcs_map.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BSI Syariah WebRTC Chatbot Server")
    parser.add_argument(
        "-t", "--test", action="store_true", default=False, help="set the server in testing mode"
    )
    parser.add_argument(
        "--host", default="localhost", help="Host for HTTP server (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=7860, help="Port for HTTP server (default: 7860)"
    )
    args = parser.parse_args()

    app.state.testing = args.test

    logger.info(f"Starting BSI Syariah WebRTC server on {args.host}:{args.port}")
    logger.info(f"Visit http://{args.host}:{args.port} to start voice chatting!")
    
    uvicorn.run(app, host=args.host, port=args.port) 