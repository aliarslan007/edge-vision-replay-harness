"""A2 replay harness: frame sources, wall-clock replay, pipeline contract, scoring and plots."""
from .schema import ATTRS, COLORS, PRESENCE_ATTRS, Frame, parse_vlm_output, validate_person
from .events import EventLog, read_log
from .sources import open_source, VideoFileSource, ImageSequenceSource, CameraSource
from .replay import run_replay, PipelineContext
from .pipelines import QueuedVLMPipeline, StubVLM, LlamaServerVLM
from .scoring import score, build_visits, summary_line
