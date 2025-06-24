#!/usr/bin/env python3
#
# Copyright (c) 2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import argparse
import asyncio
import os
from contextlib import asynccontextmanager
from typing import Dict

import uvicorn
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from bot_webrtc_clean import run_bot
from pipecat.transports.network.webrtc_connection import IceServer, SmallWebRTCConnection

load_dotenv(override=True)

app = FastAPI()

# Store connections by pc_id
pcs_map: Dict[str, SmallWebRTCConnection] = {}

ice_servers = [
    IceServer(
        urls="stun:stun.l.google.com:19302",
    )
]

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")


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

        # Run bot function with SmallWebRTC connection
        background_tasks.add_task(run_bot, pipecat_connection, app.state.testing)

    answer = pipecat_connection.get_answer()
    # Updating the peer connection inside the map
    pcs_map[answer["pc_id"]] = pipecat_connection

    return answer


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