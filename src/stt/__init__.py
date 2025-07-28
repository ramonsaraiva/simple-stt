"""
Speech-to-Text System with LLM Enhancement

A production-ready speech-to-text system with voice activity detection,
Whisper transcription, and LLM text refinement.
"""

__version__ = "1.0.0"
__author__ = "STT Team"

from .services.orchestrator import STTOrchestrator

__all__ = ["STTOrchestrator"]