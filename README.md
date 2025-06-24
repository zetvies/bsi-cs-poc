# BSI Syariah WebRTC Chatbot - Render Deployment

This is a deployment-ready package for the BSI Syariah WebRTC voice chatbot, optimized for deployment on Render.

## Features

- **Voice Chat Interface**: Real-time voice conversations with AI assistant
- **Call Mode Selection**: Inbound, Outbound, and Free conversation modes
- **Call Analytics Dashboard**: Comprehensive summary of all conversations
- **Sentiment Analysis**: AI-powered conversation sentiment scoring
- **Lead Intent Detection**: Automatic detection of customer interests
- **Sharia Compliance Monitoring**: Ensures conversations align with Islamic banking principles

## Quick Start

### 1. Environment Variables

Copy `env.example` to `.env` and configure:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Supabase Configuration
SUPABASE_URL=https://fowdecmskhvfgmdwwnxg.supabase.co
SUPABASE_KEY=your_supabase_service_role_key_here

# Server Configuration
HOST=0.0.0.0
PORT=8000
```

### 2. Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python server_webrtc.py
```

### 3. Docker Deployment

```bash
# Build the image
docker build -t bsi-syariah-chatbot .

# Run the container
docker run -p 8000:8000 --env-file .env bsi-syariah-chatbot
```

## Render Deployment

### 1. Connect Repository

1. Fork or clone this repository to your GitHub account
2. Connect your repository to Render

### 2. Create Web Service

1. Go to Render Dashboard
2. Click "New +" → "Web Service"
3. Connect your GitHub repository
4. Configure the service:

**Build Command:**
```bash
pip install -r requirements.txt
```

**Start Command:**
```bash
uvicorn server_webrtc:app --host 0.0.0.0 --port $PORT
```

### 3. Environment Variables

Add these environment variables in Render:

- `OPENAI_API_KEY`: Your OpenAI API key
- `SUPABASE_URL`: Your Supabase project URL
- `SUPABASE_KEY`: Your Supabase service role key
- `PORT`: Render will set this automatically

### 4. Deploy

Click "Create Web Service" and wait for deployment to complete.

## Application Structure

```
render-deployment/
├── server_webrtc.py          # Main FastAPI server
├── bot_webrtc_clean.py       # Bot logic and conversation handling
├── static/
│   ├── summary.html          # Analytics dashboard
│   └── custom_ui.html        # Voice chat interface
├── requirements.txt          # Python dependencies
├── Dockerfile               # Container configuration
├── .dockerignore           # Docker build exclusions
├── env.example             # Environment variables template
└── README.md               # This file
```

## API Endpoints

- `GET /` - Summary dashboard (default page)
- `GET /client/` - Voice chat interface
- `POST /api/offer` - WebRTC connection handling

## Call Modes

### Inbound Mode
- Greeting: "Assalamualaikum, selamat datang di layanan customer service BSI Syariah..."
- Purpose: Handle incoming customer inquiries

### Outbound Mode
- Greeting: "Assalamualaikum, perkenalkan saya Melina dari customer service BSI Syariah..."
- Purpose: Proactive product introduction and lead generation

### Free Mode
- Greeting: "Assalamualaikum, perkenalkan saya Melina dari customer service BSI Syariah..."
- Purpose: Open conversation for general inquiries

## Analytics Features

The summary dashboard provides:

- **Conversation History**: Complete chat transcripts
- **Sentiment Analysis**: AI-powered sentiment scoring (0-100)
- **Lead Intent Detection**: Automatic product interest identification
- **Sharia Compliance**: Monitoring for Islamic banking principles
- **Call Scoring**: Weighted scoring system (Sentiment 40% + Compliance 30% + Lead 30%)
- **Follow-up Recommendations**: AI-generated actionable next steps

## Troubleshooting

### Common Issues

1. **WebRTC Connection Failed**
   - Check browser permissions for microphone access
   - Ensure HTTPS is enabled (required for WebRTC)

2. **Audio Not Working**
   - Verify microphone permissions
   - Check browser console for errors
   - Test with different browsers

3. **Environment Variables**
   - Ensure all required environment variables are set
   - Check API key validity

### Logs

Check Render logs for detailed error information and debugging.

## Support

For technical support or questions about the BSI Syariah chatbot deployment, please refer to the main project documentation or contact the development team. 