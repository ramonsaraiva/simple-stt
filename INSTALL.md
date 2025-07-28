# Installation Guide

## Quick Start (Development)

For development or testing the refactored version:

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run using the legacy entry point:**
   ```bash
   python main.py --help
   ```

3. **Or run using the new CLI module:**
   ```bash
   python -m stt.cli --help
   ```

## Production Installation

For production use, install as a proper Python package:

1. **Install in development mode:**
   ```bash
   pip install -e .
   ```

2. **Use the `stt` command:**
   ```bash
   stt --help
   stt --tune
   stt --profile todo
   ```

## Configuration

The system will look for configuration in these locations (in order):
1. `config.yaml` (current directory)
2. `~/.stt/config.yaml` (user home)
3. `/etc/stt/config.yaml` (system-wide)

If no config file is found, defaults will be used.

## Key Improvements in Refactored Version

- **Modular architecture:** Clean separation of concerns
- **Better error handling:** Specific exceptions with clear messages
- **Type hints:** Improved code maintainability
- **Proper logging:** Structured logging with different levels
- **Configuration management:** Flexible config with validation
- **Event-driven UI:** Decoupled UI updates
- **Backward compatibility:** Old scripts still work

## Migrating from Old Version

If you have scripts using the old flat structure:

**Old way:**
```python
from audio_recorder import AudioRecorder
from config import Config
```

**New way:**
```python
from stt.audio import AudioRecorder
from stt.config import ConfigManager
```

Or use the high-level orchestrator:
```python
from stt import STTOrchestrator

orchestrator = STTOrchestrator()
orchestrator.run_transcription()
```