"""
Interruptible bot using SmallWebRTCTransport.
Based on Pipecat's 07-interruptible.py example.
"""

import os
import sys

from dotenv import load_dotenv
from loguru import logger

from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import (
    TranscriptionFrame,
    StartInterruptionFrame,
    StopInterruptionFrame,
    LLMTextFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frameworks.rtvi import RTVIConfig, RTVIObserver, RTVIProcessor
from kokoro_tts import KokoroTTSService
from pipecat.services.whisper.stt import WhisperSTTServiceMLX, MLXModel
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.base_transport import TransportParams
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import create_transport
from pipecat.processors.frame_processor import FrameProcessor

load_dotenv()

logger.remove()
logger.add(sys.stderr, level="DEBUG")


class TranscriptionFrameFixer(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            if not frame.user_id:
                frame.user_id = ""
        await self.push_frame(frame, direction)


class ChannelAnalysisStripper(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._last_frame_was_channel = False

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        # Check if this is an LLMTextFrame with "analysis" content
        if isinstance(frame, LLMTextFrame):
            if frame.text == "<|channel|>":
                self._last_frame_was_channel = True
                return
            else:
                if frame.text == "analysis" and self._last_frame_was_channel:
                    self._last_frame_was_channel = False
                    return
            self._last_frame_was_channel = False

        await self.push_frame(frame, direction)


async def run_bot(transport):
    """Main bot function that creates and runs the pipeline."""

    rtvi = RTVIProcessor(config=RTVIConfig(config=[]))

    stt = WhisperSTTServiceMLX(model=MLXModel.LARGE_V3_TURBO_Q4)

    tts = KokoroTTSService(model="prince-canuma/Kokoro-82M", voice="af_heart", sample_rate=24000)

    # Initialize LLM service
    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="http://localhost:8080/v1",
        model="",
    )

    tf_fixer = TranscriptionFrameFixer()
    channel_stripper = ChannelAnalysisStripper()

    # System prompt
    messages = [
        {
            "role": "system",
            "content": "You are a helpful and friendly AI assistant. You love to chat about life, answer questions, and help people. Keep your responses concise and natural.",
        }
    ]

    # Create context aggregator
    context = OpenAILLMContext(messages)
    context_aggregator = llm.create_context_aggregator(context)

    # Create pipeline
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            tf_fixer,
            rtvi,  # Add RTVI processor for transcription events
            context_aggregator.user(),
            llm,
            channel_stripper,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    # Create task with RTVI observer
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            allow_interruptions=True,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
        observers=[RTVIObserver(rtvi)],
    )

    @rtvi.event_handler("on_client_ready")
    async def on_client_ready(rtvi):
        await rtvi.set_bot_ready()
        # Kick off the conversation
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    @rtvi.event_handler("on_client_message")
    async def on_client_message(rtvi, message):
        """Handle custom messages from the client."""
        logger.info(f"Received client message: {message}")

        # Extract message type and data from RTVIClientMessage object
        msg_type = message.type
        msg_data = message.data if hasattr(message, "data") else {}

        if msg_type == "custom-message":
            text = msg_data.get("text", "") if isinstance(msg_data, dict) else ""
            if text:
                # Process the text message as user input
                logger.info(f"Processing custom message: {text}")
                # Send the text as a TranscriptionFrame which will be processed by the context aggregator
                await task.queue_frames(
                    [
                        StartInterruptionFrame(),
                        TranscriptionFrame(
                            text=text,
                            user_id="text-input",
                            timestamp="",
                        ),
                        StopInterruptionFrame(),
                    ]
                )

                # Send acknowledgment back to client
                await rtvi.send_server_message(
                    {"type": "message-received", "text": f"Received: {text}"}
                )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        """Handle new connection."""
        logger.info("Client connected")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        """Handle disconnection."""
        logger.info("Client disconnected")
        await task.cancel()
        logger.info("Bot stopped")

    # Create runner and run the task
    runner = PipelineRunner(handle_sigint=False)
    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with standard bot starters, including Pipecat Cloud."""

    transport_params = {
        "daily": lambda: DailyParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
        "twilio": lambda: FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
        ),
        "webrtc": lambda: TransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(),
            # turn_analyzer=LocalSmartTurnAnalyzerV2(
            #     smart_turn_model_path=None, params=SmartTurnParams()
            # ),
        ),
    }

    transport = await create_transport(runner_args, transport_params)
    await run_bot(transport)


if __name__ == "__main__":
    from pipecat.runner.run import main

    main()
