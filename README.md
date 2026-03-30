# 🎙️ AI Talks — Gemini 3.1 Flash Live (Dual-Agent Podcast)

📺 **Watch the Demo:** [AI Talks — Gemini 3.1 Flash Live Podcast](https://youtu.be/EbJ2NeRcJtk)

**🍏 macOS Only (Out of the Box):** This project relies on the [BlackHole](https://existential.audio/blackhole/) virtual audio driver for native zero-latency audio routing, which is specific to macOS. *(To run this on Windows, you will need to adapt the `INPUT_DEVICE` and `OUTPUT_DEVICE` in the script to point to an alternative virtual audio router like [VB-Cable](https://vb-audio.com/Cable/) or Virtual Audio Cable).*

A recursive audio loop between two independent Google Gemini 3.1 Flash Live instances. One AI's voice output becomes the other AI's physical microphone input, creating a real, unscripted, native voice conversation.


## ⚡ What is Gemini 3.1 Flash Live?

Released in late March 2026, `gemini-3.1-flash-live-preview` is a highly specialized, ultra-low-latency model built specifically for the Gemini Live API. 

Unlike previous models where AI generates text that gets read out loud by a separate TTS (Text-to-Speech) system, Flash 3.1 is **Native Audio-to-Audio (A2A)**. It ingests raw audio (even sensing tone, sarcasm, and pauses) and streams raw audio right back. 

**Model Trade-offs:**
- Built purely for speed (sports car approach).
- Smaller 131k token context window.
- Drops heavy backend features (No massive JSON extraction or batch execution).
- Keeps the interactive essentials: Synchronous Function Calling and Google Search Grounding.

## 🚀 Quick Start Guide

### Step 1: Install BlackHole Audio Driver
A virtual audio cable is required to link the AI voices natively.
1. Open Terminal and install via Homebrew: `brew install blackhole-2ch`
   *(This requires an active terminal and your administrator password).*
2. **Troubleshooting:** If BlackHole doesn't immediately appear in your system audio settings, you can refresh the CoreAudio registry by running `sudo killall coreaudiod`.

### Step 2: Configure a Multi-Output Device
For the AIs to hear each other natively *and* for you to monitor the conversation, you need to split the audio signal.
1. Press `Cmd + Space` and open **Audio MIDI Setup**.
2. Click the `+` icon in the bottom-left corner and select **Create Multi-Output Device**.
3. Check the boxes for both **BlackHole 2ch** AND your **MacBook Speakers** (or headphones).

### Step 3: Install Dependencies
```bash
pip install google-genai>=1.0.0 websockets>=13.0 rich>=13.0 numpy>=1.26.0 python-dotenv>=1.0.0 sounddevice>=0.4.6
```

### Step 4: Environment Context
Create a `.env` file in the project folder with your API key:
```env
GEMINI_API_KEY=your_api_key_here
```

### Step 5: Run the Orchestrator
```bash
python orchestrator.py
```
The script loads the provided personas (`gary.json` and `brenda.json`) and prompts you for an interactive voice discussion.

## 📁 Repository Structure

```text
├── orchestrator.py              # Dual-agent audio router 
├── requirements.txt             # Python dependencies
├── gary.json                    # Gary — oversharing IT help desk guy
├── brenda.json                  # Brenda — unimpressable retired HR manager
├── .env.example                 # Example environment template
└── LICENSE                      # Apache 2.0 License
```

## ⚖️ License

This project is licensed under the **Apache License 2.0**. See the [LICENSE](LICENSE) file for complete details.
