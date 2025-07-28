# Speech-to-Text System for Hyprland

> **Disclaimer**: This codebase was fully built using AI language models.

A production-ready Python speech-to-text system that integrates with Hyprland. Features voice activity detection, parallel processing, and LLM text enhancement with a clean modular architecture.

## ✨ Features

- **🎙️ Voice Activity Detection**: Waits for you to start speaking before beginning silence timeout
- **⚡ Parallel Processing**: Model loading happens in background while you speak - no waiting!
- **🤖 High-Quality STT**: Uses OpenAI Whisper for accurate speech recognition with GPU acceleration  
- **✨ AI Text Refinement**: Optional text cleanup and formatting using OpenAI GPT or Anthropic Claude
- **📋 Clipboard Integration**: Automatically copies refined text to clipboard or auto-pastes
- **🎯 Auto-Tuning**: Built-in threshold calibration for optimal voice detection
- **🎨 Tokyo Night UI**: Beautiful dark theme with real-time status updates
- **📝 Comprehensive Logging**: Structured logging with different levels
- **🔧 Flexible Configuration**: YAML config with environment variable support
- **🏗️ Production Ready**: Modular architecture with proper error handling

## 🚀 Quick Start

### Installation

**Option 1: Development/Testing**
```bash
# Clone and install dependencies
git clone <your-repo>
cd stt
pip install -r requirements.txt

# Install system dependencies
sudo pacman -S xdotool  # Arch Linux
# or
sudo apt install xdotool  # Ubuntu/Debian
```

**Option 2: Production Installation**
```bash
# Install as proper Python package
pip install -e .

# Now use the 'stt' command anywhere
stt --help
```

### Configuration

1. **Set up API Keys** - Create a `.env` file:
   ```bash
   OPENAI_API_KEY=your-openai-key-here
   ANTHROPIC_API_KEY=your-anthropic-key-here
   ```

2. **Tune Voice Detection**:
   ```bash
   # For development
   python main.py --tune
   
   # For production install
   stt --tune
   ```
   Follow the prompts to calibrate voice detection for your microphone.

3. **Configure Hyprland** - Add to `~/.config/hypr/hyprland.conf`:
   ```bash
   # Development usage
   bind = ALT SHIFT, R, exec, cd /path/to/stt && python main.py
   bind = ALT SHIFT, T, exec, cd /path/to/stt && python main.py --profile todo
   
   # Production usage (after pip install -e .)
   bind = ALT SHIFT, R, exec, stt
   bind = ALT SHIFT, T, exec, stt --profile todo
   ```

## 📖 Usage

### Basic Usage

**Development:**
```bash
python main.py                    # Start recording with default settings
python main.py --profile todo     # Use 'todo' LLM profile  
python main.py --no-llm           # Skip LLM refinement
python main.py --verbose          # Enable debug logging
```

**Production:**
```bash
stt                               # Start recording
stt --profile todo                # Use specific profile
stt --list-profiles               # Show available profiles
stt --tune                        # Calibrate voice detection
stt --config /path/to/config.yaml # Use custom config
```

### How It Works

1. **🎤 Start**: Press your keybind → UI appears showing "Waiting for voice..."
2. **🗣️ Speak**: Start speaking → UI changes to "Recording" with timer
3. **🤫 Silence**: Stop speaking → After 2 seconds of silence, recording stops
4. **⚡ Process**: Whisper transcribes → LLM enhances (optional) → Copies to clipboard
5. **✅ Done**: "Copied to clipboard!" → UI auto-hides

### Available Profiles

Run `stt --list-profiles` to see all available LLM profiles:

- **general**: General text cleanup and grammar correction
- **todo**: Convert speech to actionable todo items
- **email**: Format as professional email
- **slack**: Format for Slack/chat messages
- **obsidian**: Format as Obsidian markdown notes
- **code_comment**: Format as code documentation

## ⚙️ Configuration

The system looks for configuration in this order:
1. `config.yaml` (current directory)
2. `~/.stt/config.yaml` (user home) 
3. `/etc/stt/config.yaml` (system-wide)

### Key Settings

```yaml
audio:
  silence_threshold: 20           # Voice detection threshold (tune with --tune)
  silence_duration: 2.0           # Seconds of silence before stopping
  max_recording_time: 120.0       # Maximum recording time (safety)

whisper:
  model: turbo                    # tiny, base, small, medium, large, turbo
  language: en                    # Language code or 'auto'
  device: cuda                    # cuda, cpu, or auto
  compute_type: float16           # float16, int8, or float32

llm:
  enabled: true                   # Enable/disable LLM refinement
  provider: openai                # openai or anthropic  
  model: gpt-3.5-turbo           # Model for text enhancement
  default_profile: general        # Default enhancement profile

ui:
  enabled: true                   # Show UI overlay
  auto_hide_delay: 3.0           # Seconds before auto-hide

clipboard:
  auto_paste: false               # Auto-paste to active window
  paste_delay: 0.1               # Delay before pasting
```

### Environment Variables

Load from `.env` file or set directly:
```bash
OPENAI_API_KEY=your_key_here
ANTHROPIC_API_KEY=your_key_here
```

## 🏗️ Architecture

The refactored system uses a clean modular architecture:

```
src/stt/
├── audio/          # Voice recording and detection
├── speech/         # Transcription and enhancement  
├── ui/             # Tokyo Night themed overlay
├── config/         # Configuration management
├── services/       # Core business logic
└── cli.py          # Command-line interface
```

### Key Components

- **🎙️ AudioRecorder**: Voice activity detection and recording
- **🤖 SpeechTranscriber**: Whisper integration with async loading
- **✨ TextEnhancer**: LLM text refinement with multiple providers
- **🎨 UIManager**: Tokyo Night themed status overlay
- **⚙️ ConfigManager**: Flexible configuration with validation
- **🎯 STTOrchestrator**: Main service coordinator

## 🔧 Development

### Adding New LLM Profiles

Edit `config.yaml`:
```yaml
llm:
  profiles:
    my_profile:
      name: "My Custom Profile"
      prompt: "Your custom prompt here..."
```

### Adding New LLM Providers

Extend `TextEnhancer` in `src/stt/speech/enhancer.py`:
```python
def _call_my_provider(self, prompt: str, text: str) -> Optional[str]:
    # Implementation here
    pass
```

## 🐛 Troubleshooting

### Audio Issues
- **No recording**: Check microphone permissions and audio drivers
- **False voice detection**: Run `stt --tune` to recalibrate threshold
- **Always waiting for voice**: Threshold too high, run `--tune` or lower `silence_threshold`

### API Issues  
- **API key errors**: Check `.env` file exists and has correct keys
- **Rate limiting**: Check API quotas and billing
- **Provider errors**: Try switching between `openai` and `anthropic`

### System Issues
- **UI not showing**: Install tkinter: `sudo apt install python3-tk`
- **Auto-paste not working**: Install xdotool: `sudo apt install xdotool`
- **Import errors**: Make sure you've installed dependencies: `pip install -r requirements.txt`

### Debugging
```bash
# Enable verbose logging
stt --verbose

# Check configuration
stt --list-profiles

# Test API connectivity (will attempt to enhance sample text)
echo "test text" | stt --profile general --verbose
```

## 📝 Logging

Logs are written to both console and `stt.log`:
- **INFO**: General operation status
- **DEBUG**: Detailed debugging (use `--verbose`)  
- **ERROR**: Error conditions with context
- **WARNING**: Important notices

## 🔄 Migration from Old Version

The system maintains backward compatibility:

```bash
# Old way (still works, shows deprecation warning)
python main.py --profile todo

# New way (recommended)
stt --profile todo
```

For scripts using the old import structure, update:
```python
# Old
from audio_recorder import AudioRecorder

# New  
from stt.audio import AudioRecorder
# or
from stt import STTOrchestrator
```

## 🤝 Contributing

This is a production-ready system with proper:
- ✅ Modular architecture
- ✅ Error handling and logging  
- ✅ Type hints and documentation
- ✅ Configuration management
- ✅ Backward compatibility

Feel free to extend with new providers, profiles, or features!