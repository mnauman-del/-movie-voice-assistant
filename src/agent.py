import logging
import asyncio
import base64
from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    inference,
    room_io,
)
import os
from livekit.plugins import groq
from livekit.agents import function_tool, RunContext
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel
from livekit.plugins import deepgram
from livekit.agents.metrics import LLMMetrics, STTMetrics, TTSMetrics, EOUMetrics
from livekit.agents.telemetry import set_tracer_provider
import aiohttp

logger = logging.getLogger("agent")

load_dotenv(".env.local")

os.environ["MOVIES_READ_ACCESS_TOKEN"] = os.getenv("MOVIES_READ_ACCESS_TOKEN")
os.environ["DEEPGRAM_API_KEY"] = os.getenv("DEEPGRAM_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {os.getenv('MOVIES_READ_ACCESS_TOKEN')}",
}

async def fetch_data(movieName: str, type: str = "movie"):
    params = {
        "query": movieName,
        "include_adult": "false",
        "language": "en-US",
        "page": 1,
    }
    logger.info(f"Test fetch movie params: {params}")
    async with aiohttp.ClientSession() as session:
        async with session.get(
            f"https://api.themoviedb.org/3/search/{type}",
            headers=headers,
            params=params,
        ) as response:
            return await response.json()

   

def setup_langfuse(
    host: str | None = None, public_key: str | None = None, secret_key: str | None = None
):
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
 
    public_key = public_key or os.getenv("LANGFUSE_PUBLIC_KEY")
    secret_key = secret_key or os.getenv("LANGFUSE_SECRET_KEY")
    host = host or os.getenv("LANGFUSE_BASE_URL")
 
    if not public_key or not secret_key or not host:
        raise ValueError("LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_BASE_URL must be set")
 
    langfuse_auth = base64.b64encode(f"{public_key}:{secret_key}".encode()).decode()
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = f"{host.rstrip('/')}/api/public/otel"
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"Authorization=Basic {langfuse_auth}"
 
    trace_provider = TracerProvider()
    trace_provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    set_tracer_provider(trace_provider)


class Assistant(Agent):
    def __init__(self, stt, llm, tts) -> None:
        super().__init__(
            instructions="""You are a helpful voice AI assistant. The user is interacting with you via voice, even if you perceive the conversation as text. You only help in the domain of movies and film trivia.You know nothing outside of it. You assume every question is about movies and tv shows.
            You eagerly assist users with their questions by providing information from your extensive knowledge.
            Your responses are concise, to the point, and without any complex formatting or punctuation including emojis, asterisks, or other symbols.
            You are curious, friendly, and have a sense of humor.
            
            YOU ARE ONLY ALLOWED TO USE THE TOOLS PROVIDED TO YOU. DONT SEARCH ON THE INTERNET OUTSIDE OF THE TOOLS.""",
        )
   
    @function_tool 
    async def lookup_tv_shows(self, context: RunContext, showName: str, releaseYear: str = None):
        """Use this tool to look up information about TV shows.

        Args:
            showName: The TV show title to search for.
            releaseYear: (Optional) The release year of the TV show to narrow down results.
        """

        logger.info(f"Looking up TV show for {showName}")

        res = await fetch_data(showName, type="tv")
        logger.info(f"TV show lookup result raw: {res}")

        results = res.get('results', [])

        if not results:
            return f"Sorry, I couldn't find any information about '{showName}'."
        
        for item in results:
            year = item.get('first_air_date', '')[:4] 
            if releaseYear:
                if year == releaseYear:
                    res = item
                    break
        else:
            res = results[0]  
    
        logger.info(f"TV show lookup result: {res}")

        return f"Information about the TV show '{showName}': {res}"


    @function_tool
    async def lookup_movies(self, context: RunContext, movieName: str, releaseYear: str = None):
        """Use this tool to look up information about movies.

        Args:
            movieName: The movie title to search for.
            releaseYear: (Optional) The release year of the movie to narrow down results.
        """

        logger.info(f"Looking up movie for {movieName}")

        res = await fetch_data(movieName)
        logger.info(f"Movie lookup result raw: {res}")

        results = res.get('results', [])

        if not results:
            return f"Sorry, I couldn't find any information about '{movieName}'."
        
        for item in results:
            year = item.get('release_date', '')[:4] 
            if releaseYear:
                if year == releaseYear:
                    res = item
                    break
        else:
            res = results[0]  
    
        logger.info(f"Movie lookup result: {res}")

        return f"Information about the movie '{movieName}': {res}"


server = AgentServer()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


@server.rtc_session()
async def my_agent(ctx: JobContext):
    # Logging setup
    # Add any other context you want in all log entries here
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    # Set up a voice AI pipeline using OpenAI, Cartesia, AssemblyAI, and the LiveKit turn detector
    stt = deepgram.STTv2(model="flux-general-en", eager_eot_threshold=0.4)
    llm = groq.LLM(model="openai/gpt-oss-120b")
    tts = deepgram.TTS(model="aura-asteria-en")
    turn_detection = MultilingualModel()
    vad = ctx.proc.userdata["vad"]


    session = AgentSession(
        # Speech-to-text (STT) is your agent's ears, turning the user's speech into text that the LLM can understand
        # See all available models at https://docs.livekit.io/agents/models/stt/
        stt=stt,
        # A Large Language Model (LLM) is your agent's brain, processing user input and generating a response
        # See all available models at https://docs.livekit.io/agents/models/llm/
        llm=llm,
        # Text-to-speech (TTS) is your agent's voice, turning the LLM's text into speech that the user can hear
        # See all available models as well as voice selections at https://docs.livekit.io/agents/models/tts/
        tts=tts,
        # VAD and turn detection are used to determine when the user is speaking and when the agent should respond
        # See more at https://docs.livekit.io/agents/build/turns
        turn_detection=turn_detection,
        vad=vad,
        # allow the LLM to generate a response while waiting for the end of turn
        # See more at https://docs.livekit.io/agents/build/audio/#preemptive-generation
        preemptive_generation=True,
    )

    # To use a realtime model instead of a voice pipeline, use the following session setup instead.
    # (Note: This is for the OpenAI Realtime API. For other providers, see https://docs.livekit.io/agents/models/realtime/))
    # 1. Install livekit-agents[openai]
    # 2. Set OPENAI_API_KEY in .env.local
    # 3. Add `from livekit.plugins import openai` to the top of this file
    # 4. Use the following session setup instead of the version above
    # session = AgentSession(
    #     llm=openai.realtime.RealtimeModel(voice="marin")
    # )

    # # Add a virtual avatar to the session, if desired
    # # For other providers, see https://docs.livekit.io/agents/models/avatar/
    # avatar = hedra.AvatarSession(
    #   avatar_id="...",  # See https://docs.livekit.io/agents/models/avatar/plugins/hedra
    # )
    # # Start the avatar and wait for it to join
    # await avatar.start(session, room=ctx.room)

    # Start the session, which initializes the voice pipeline and warms up the models
    await session.start(
        agent=Assistant(stt=stt, llm=llm, tts=tts),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony()
                if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                else noise_cancellation.BVC(),
            ),
        ),
    )

    # Join the room and connect to the user
    await ctx.connect()

if __name__ == "__main__":
    setup_langfuse()
    cli.run_app(server)
