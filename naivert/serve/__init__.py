"""Host-side request handling, chat formatting, token streaming, and stop detection.

These modules run on the CPU. ServerContext drives a session engine; the HTTP
worker serializes requests and publishes results without accessing CUDA directly.
"""

from .task import FinalResult, OpenedResult, PartialResult, ServerTask, TaskParams, Timings
from .slot import IllegalSlotTransition, ServerSlot, SlotEvent, SlotState
from .tokens import ServerTokens, StopMatch, StopMatcher, StopTokenSequenceMatcher, check_stop_strings

__all__ = [
    "FinalResult", "IllegalSlotTransition", "OpenedResult", "PartialResult", "ServerSlot", "ServerTask", "ServerTokens",
    "SlotEvent", "SlotState", "StopMatch", "StopMatcher", "StopTokenSequenceMatcher", "TaskParams", "Timings",
    "check_stop_strings",
]
