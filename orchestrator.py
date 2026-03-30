"""
AI Talks — Dual-Agent Orchestrator (Interactive)
===================================
Connects two Gemini 3.1 Flash Live instances in a recursive audio loop.
Using Virtual Studio Audio Routing via BlackHole and Multi-Output Device.

Usage:
    python orchestrator.py

Environment:
    GEMINI_API_KEY — Your Google AI API key
"""

import asyncio
import json
import os
import sys
import time
import queue
import threading
import signal
from pathlib import Path
from datetime import datetime
from typing import Optional

# Load local .env file
try:
    from dotenv import load_dotenv
    _script_dir = Path(__file__).parent
    load_dotenv(_script_dir / ".env")
except ImportError:
    pass

import sounddevice as sd
from google import genai
from google.genai import types
from rich.console import Console
from rich.panel import Panel

# ─── Constants ─────────────────────────────────────────────────────────────────

INPUT_SAMPLE_RATE = 16000
OUTPUT_SAMPLE_RATE = 24000
CHANNELS = 1
SAMPLE_WIDTH = 2
MODEL = "gemini-3.1-flash-live-preview"

INPUT_DEVICE = "BlackHole 2ch"
OUTPUT_DEVICE = "Multi-Output Device"

# Timings
WRAP_UP_SECONDS = 60     # Send wrap-up instruction to Agent 1
SILENCE_TIMEOUT = 8.0    # Shut down if 8s of silence occurs after wrap-up
MAX_HARD_TIMEOUT = 180   # Absolute fallback limit (3 mins)

# ─── Console ───────────────────────────────────────────────────────────────────

console = Console()

# ─── Agent State ───────────────────────────────────────────────────────────────

class AgentState:
    IDLE = "IDLE"
    LISTENING = "LISTENING"
    SPEAKING = "SPEAKING"
    INTERRUPTED = "INTERRUPTED"

    def __init__(self, name: str, accent_color: str, avatar_emoji: str):
        self.name = name
        self.accent_color = accent_color
        self.avatar_emoji = avatar_emoji
        self.state = self.IDLE
        self.current_text = ""
        self.transcript_lines: list[dict] = []

    def set_state(self, new_state: str, orchestrator: "Orchestrator"):
        old_state = self.state
        self.state = new_state
        if old_state != new_state:
            orchestrator.log_state_change(self, old_state, new_state)

    def append_transcript(self, text: str, orchestrator: "Orchestrator"):
        self.current_text += text

    def finalize_utterance(self, orchestrator: "Orchestrator"):
        if self.current_text.strip():
            entry = {
                "speaker": self.name,
                "text": self.current_text.strip(),
                "timestamp": time.time()
            }
            self.transcript_lines.append(entry)
            orchestrator.log_utterance(self, self.current_text.strip())
            orchestrator.last_utterance_time = time.time()  # Track silence
            orchestrator.last_speaker = self.name
        self.current_text = ""

# ─── Orchestrator ──────────────────────────────────────────────────────────────

class Orchestrator:
    def __init__(self, bootstrap_prompt: str):
        self.bootstrap_prompt = bootstrap_prompt
        
        base_dir = Path(__file__).parent
        with open(base_dir / "gary.json") as f:
            self.persona_1 = json.load(f)
        with open(base_dir / "brenda.json") as f:
            self.persona_2 = json.load(f)

        self.agent1 = AgentState(
            self.persona_1["name"],
            self.persona_1["accent_color"],
            self.persona_1["avatar_emoji"]
        )
        self.agent2 = AgentState(
            self.persona_2["name"],
            self.persona_2["accent_color"],
            self.persona_2["avatar_emoji"]
        )

        self.start_time: Optional[float] = None
        self.conversation_log: list[dict] = []
        self.running = False
        self.wrap_up_sent = False
        self.suppress_audio_from = None  # Mute this agent's audio output
        self.mics_cut = False
        self.last_utterance_time = time.time()
        self.last_speaker = self.agent1.name
        self._shutdown_event: Optional[asyncio.Event] = None  # Set in run()

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            console.print("[bold red]ERROR:[/] GEMINI_API_KEY environment variable not set.")
            sys.exit(1)
        self.client = genai.Client(api_key=api_key)

        self.audio_in_q = asyncio.Queue()
        self.audio_out_q = queue.Queue()

    def stop(self, reason: str = ""):
        """Signal all tasks to shut down immediately."""
        if not self.running:
            return
        self.running = False
        if reason:
            self.log_system(reason)
        if self._shutdown_event:
            self._shutdown_event.set()
        
    def elapsed(self) -> str:
        if self.start_time is None:
            return "00:00"
        delta = time.time() - self.start_time
        minutes = int(delta) // 60
        seconds = int(delta) % 60
        return f"{minutes:02d}:{seconds:02d}"

    def log_state_change(self, agent: AgentState, old: str, new: str):
        icons = {
            AgentState.IDLE: "💤",
            AgentState.LISTENING: "🎧",
            AgentState.SPEAKING: "🗣️",
            AgentState.INTERRUPTED: "⚡"
        }
        console.print(
            f"[dim]{self.elapsed()}[/] {icons.get(new, '❓')} "
            f"[{agent.accent_color}]{agent.name}[/] → {new}"
        )

    def log_utterance(self, agent: AgentState, text: str):
        console.print(
            f"[dim]{self.elapsed()}[/] "
            f"🗣️ [{agent.accent_color}]{agent.name}[/]: \"{text}\""
        )
        self.conversation_log.append({
            "speaker": agent.name,
            "text": text,
            "elapsed": self.elapsed(),
            "timestamp": time.time()
        })

    def log_system(self, message: str):
        console.print(f"[dim]{self.elapsed()}[/] [dim cyan]SYSTEM:[/] {message}")



    def _mic_worker(self, loop):
        try:
            with sd.RawInputStream(samplerate=INPUT_SAMPLE_RATE, channels=1, dtype='int16', device=INPUT_DEVICE, blocksize=4096) as stream:
                while self.running:
                    data, overflow = stream.read(4096)
                    if self.running and len(data) > 0:
                        loop.call_soon_threadsafe(self.audio_in_q.put_nowait, bytes(data))
        except Exception as e:
            self.log_system(f"Mic Worker Error: {e}")

    def _spk_worker(self):
        try:
            with sd.RawOutputStream(samplerate=OUTPUT_SAMPLE_RATE, channels=1, dtype='int16', device=OUTPUT_DEVICE) as stream:
                while self.running:
                    try:
                        data = self.audio_out_q.get(timeout=0.2)
                        if data:
                            stream.write(data)
                    except queue.Empty:
                        continue
        except Exception as e:
            self.log_system(f"Speaker Worker Error: {e}")

    def _make_config(self, persona: dict) -> types.LiveConnectConfig:
        return types.LiveConnectConfig(
            system_instruction=types.Content(
                parts=[types.Part.from_text(text=persona["system_instruction"])]
            ),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=persona.get("voice_name", "Aoede")
                    )
                )
            ),
            response_modalities=[types.Modality.AUDIO],
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            realtime_input_config=types.RealtimeInputConfig(
                turn_coverage="TURN_INCLUDES_ONLY_ACTIVITY",
            ),
        )

    async def _route_mic(self, session1, session2):
        while self.running:
            try:
                if self.mics_cut:
                    await asyncio.sleep(0.1)
                    continue
                chunk = await self.audio_in_q.get()
                blob = types.Blob(data=chunk, mime_type="audio/pcm;rate=16000")
                
                tasks = []
                if self.agent1.state != AgentState.SPEAKING:
                    tasks.append(session1.send_realtime_input(audio=blob))
                if self.agent2.state != AgentState.SPEAKING:
                    tasks.append(session2.send_realtime_input(audio=blob))
                
                if tasks:
                    await asyncio.gather(*tasks)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.log_system(f"Route Mic error: {e}")

    async def _receive_loop(self, agent: AgentState, session):
        try:
            while self.running:
                async for response in session.receive():
                    if not self.running:
                        break

                    server_content = response.server_content
                    if server_content is None:
                        continue

                    if server_content.interrupted:
                        agent.set_state(AgentState.INTERRUPTED, self)
                        agent.finalize_utterance(self)
                        await asyncio.sleep(0.05)
                        agent.set_state(AgentState.LISTENING, self)
                        continue

                    if server_content.turn_complete:
                        agent.finalize_utterance(self)
                        agent.set_state(AgentState.LISTENING, self)
                        if hasattr(agent, "turn_complete_event"):
                            agent.turn_complete_event.set()
                        continue

                    if server_content.model_turn and server_content.model_turn.parts:
                        for part in server_content.model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                if self.suppress_audio_from == agent.name:
                                    continue  # Muted during other agent's goodbye

                                audio_bytes = part.inline_data.data
                                agent.set_state(AgentState.SPEAKING, self)
                                self.audio_out_q.put(audio_bytes)

                    if server_content.output_transcription and server_content.output_transcription.text:
                        agent.append_transcript(server_content.output_transcription.text, self)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.log_system(f"Receive Loop ({agent.name}) error: {e}")

    async def _silence_detector(self, session1, session2):
        """Watches for organic silence. Shuts down if wrapped up, otherwise prods to continue."""
        try:
            while self.running:
                await asyncio.sleep(1)
                
                # Ignore initial connection silence/load times
                if (time.time() - self.start_time) < 15.0:
                    continue
                    
                # Check for silence logic
                if (time.time() - self.last_utterance_time) > SILENCE_TIMEOUT:
                    if self.agent1.state != AgentState.SPEAKING and self.agent2.state != AgentState.SPEAKING:
                        if self.wrap_up_sent:
                            self.stop(f"Logical podcast end detected ({SILENCE_TIMEOUT} seconds of mutual silence). Shutting down naturally.")
                            break
                        else:
                            self.log_system("Dead air detected before wrap-up. Prodding agents to continue...")
                            # Prod whoever didn't speak last
                            target_session = session1 if self.last_speaker == self.agent2.name else session2
                            
                            cue = "DIRECTOR'S NOTE: Keep the podcast going! Do not say goodbye. Ask a question or bring up a new topic!"
                            await target_session.send_realtime_input(text=cue)
                            
                            # Reset the utterance timer to avoid spamming the prod
                            self.last_utterance_time = time.time()

        except asyncio.CancelledError:
            pass

    async def _wait_for_turn(self, agent: AgentState, timeout: float = 20):
        """Wait for an agent's websocket turn to complete, AND physical audio to finish playing."""
        if hasattr(agent, "turn_complete_event"):
            try:
                await asyncio.wait_for(agent.turn_complete_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                self.log_system(f"{agent.name} turnaround timed out.")
        
        # Wait for actual physical audio block to finish draining to speakers
        while not self.audio_out_q.empty():
            await asyncio.sleep(0.1)
        
        # Pad the handoff naturally
        await asyncio.sleep(1.0)

    async def _director_cue(self, session1, session2):
        """The Producer: sequential cue → wait → cue → wait → shutdown."""
        try:
            await asyncio.sleep(WRAP_UP_SECONDS)
            if not self.running:
                return

            # ── Phase 1: Cue Gary to wrap up ──
            self.log_system(f"DIRECTOR'S CUE → {self.agent1.name} ({WRAP_UP_SECONDS}s mark)")
            self.wrap_up_sent = True
            
            if hasattr(self.agent1, "turn_complete_event"):
                self.agent1.turn_complete_event.clear()
                
            cue_agent1 = (
                "DIRECTOR'S NOTE: The podcast is out of time. "
                "Wrap up your current thought in one sentence, "
                "thank the other person, and say a final goodbye."
            )
            await session1.send_realtime_input(text=cue_agent1)

            # Wait for Gary to speak his goodbye and finish
            await self._wait_for_turn(self.agent1)

            if not self.running:
                return

            # ── Phase 2: Cut mics, mute Gary, cue Brenda ──
            self.mics_cut = True
            self.suppress_audio_from = self.agent1.name
            self.log_system(f"Mics cut. DIRECTOR'S CUE → {self.agent2.name}")
            
            if hasattr(self.agent2, "turn_complete_event"):
                self.agent2.turn_complete_event.clear()
            
            cue_agent2 = (
                "DIRECTOR'S NOTE: Say a brief, warm goodbye to end the episode."
            )
            await session2.send_realtime_input(text=cue_agent2)

            # Wait for Brenda to speak her goodbye and finish
            await self._wait_for_turn(self.agent2)

            # Both agents signed off — clean shutdown
            self.stop("Both agents signed off. Ending podcast.")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.log_system(f"Director Cue Error: {e}")

    async def run(self):
        self.running = True
        self.start_time = time.time()
        self._shutdown_event = asyncio.Event()
        self.last_utterance_time = time.time()
        
        self.agent1.turn_complete_event = asyncio.Event()
        self.agent2.turn_complete_event = asyncio.Event()

        loop = asyncio.get_running_loop()
        self._mic_thread = threading.Thread(target=self._mic_worker, args=(loop,), daemon=True)
        self._spk_thread = threading.Thread(target=self._spk_worker, daemon=True)
        self._mic_thread.start()
        self._spk_thread.start()

        try:
            config1 = self._make_config(self.persona_1)
            config2 = self._make_config(self.persona_2)

            self.log_system(f"Connecting {self.agent1.name}...")
            async with self.client.aio.live.connect(model=MODEL, config=config1) as session1:
                self.agent1.set_state(AgentState.IDLE, self)
                self.log_system(f"{self.agent1.name} connected ✓")

                self.log_system(f"Connecting {self.agent2.name}...")
                async with self.client.aio.live.connect(model=MODEL, config=config2) as session2:
                    self.agent2.set_state(AgentState.IDLE, self)
                    self.log_system(f"{self.agent2.name} connected ✓")
                    console.print()

                    self.log_system(f"Bootstrapping {self.agent1.name} with your prompt...")
                    await session1.send_realtime_input(text=self.bootstrap_prompt)
                    self.agent1.set_state(AgentState.SPEAKING, self)
                    self.agent2.set_state(AgentState.LISTENING, self)

                    console.print("\n[bold red]🔴 LIVE — Conversation started![/]\n")

                    mic_routing_task = asyncio.create_task(self._route_mic(session1, session2))
                    recv1_task = asyncio.create_task(self._receive_loop(self.agent1, session1))
                    recv2_task = asyncio.create_task(self._receive_loop(self.agent2, session2))
                    silence_monitor_task = asyncio.create_task(self._silence_detector(session1, session2))
                    director_task = asyncio.create_task(self._director_cue(session1, session2))

                    all_tasks = [mic_routing_task, recv1_task, recv2_task, silence_monitor_task, director_task]

                    try:
                        # Wait for shutdown signal OR hard timeout — whichever comes first
                        await asyncio.wait_for(
                            self._shutdown_event.wait(),
                            timeout=MAX_HARD_TIMEOUT
                        )
                    except asyncio.TimeoutError:
                        self.log_system(f"Hard fallback hit ({MAX_HARD_TIMEOUT}s). Forcing end.")
                    except asyncio.CancelledError:
                        self.log_system("Conversation cancelled.")
                    finally:
                        self.running = False
                        for t in all_tasks:
                            t.cancel()
                        # Give tasks a moment to handle cancellation
                        await asyncio.gather(*all_tasks, return_exceptions=True)

        except Exception as e:
            console.print(f"[bold red]Error:[/] {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.running = False
            console.print()
            self.log_system("Conversation ended.")

            self.log_system("Waiting for audio hardware buffer cache flush...")
            time.sleep(1.0)

            console.print()
            total_utterances = len(self.conversation_log)
            console.print(Panel(
                f"[bold]Episode Complete[/]\n\n"
                f"Duration: {self.elapsed()}\n"
                f"Total utterances: {total_utterances}\n"
                f"{self.agent1.name}: {sum(1 for e in self.conversation_log if e['speaker'] == self.agent1.name)} turns\n"
                f"{self.agent2.name}: {sum(1 for e in self.conversation_log if e['speaker'] == self.agent2.name)} turns",
                title="[bold cyan]SUMMARY[/]",
                border_style="dim"
            ))


def main():
    base_dir = Path(__file__).parent
    
    # Load Personas to display
    with open(base_dir / "gary.json") as f:
        persona_1 = json.load(f)
    with open(base_dir / "brenda.json") as f:
        persona_2 = json.load(f)

    # Print Persona Definitions
    console.print()
    console.print(Panel(
        f"[bold cyan]Agent 1: {persona_1['name']}[/]\n[italic]{persona_1['system_instruction']}[/]",
        title="[bold]Persona Loaded[/]",
        border_style="cyan",
        padding=(1, 2)
    ))
    console.print(Panel(
        f"[bold yellow]Agent 2: {persona_2['name']}[/]\n[italic]{persona_2['system_instruction']}[/]",
        title="[bold]Persona Loaded[/]",
        border_style="yellow",
        padding=(1, 2)
    ))

    # Ask for Interactive Prompt
    console.print()
    console.print(f"[bold green]Enter the starting prompt for {persona_1['name']}: [/]")
    
    try:
        bootstrap_prompt = input("> ")
    except KeyboardInterrupt:
        console.print("\n[yellow]Cancelled.[/]")
        sys.exit(0)

    if not bootstrap_prompt.strip():
        bootstrap_prompt = "Hey there, introduce yourself!"
        console.print(f"[dim]Using default prompt: '{bootstrap_prompt}'[/]")

    console.print("\n[dim]Initializing...[/]\n")

    orchestrator = Orchestrator(bootstrap_prompt)

    def signal_handler(sig, frame):
        console.print("\n[yellow]Ctrl+C detected — stopping...[/]")
        orchestrator.stop()

    signal.signal(signal.SIGINT, signal_handler)
    asyncio.run(orchestrator.run())

if __name__ == "__main__":
    main()
