# Speech-to-Text System for Hyprland

> **Disclaimer**: This codebase was fully built using AI language models.

A Python-based speech-to-text system that integrates with Hyprland. Trigger recording with a keybinding, speak your message, and the system will automatically transcribe your speech, refine it with an LLM, and copy it to your clipboard.

## Features

- **Parallel Processing**: Model loading happens in background while you speak - no waiting!
- **Silence Detection**: Automatically stops recording when you stop speaking
- **High-Quality STT**: Uses OpenAI Whisper for accurate speech recognition with GPU acceleration
- **AI Text Refinement**: Optional text cleanup and formatting using OpenAI GPT or Anthropic Claude
- **Clipboard Integration**: Automatically copies refined text to clipboard
- **Auto-Paste**: Optional automatic pasting into the active window
- **Auto-Tuning**: Built-in threshold calibration for optimal silence detection
- **Comprehensive Logging**: Detailed logging to file and console
- **YAML Configuration**: Easy-to-edit configuration format

## Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   
   # Install system dependencies
   sudo pacman -S xdotool  # For Arch Linux
   # or
   sudo apt install xdotool  # For Ubuntu/Debian
   ```

2. **Configure API Keys**:
   Create a `.env` file:
   ```bash
   OPENAI_API_KEY=your-openai-key-here
   ANTHROPIC_API_KEY=your-anthropic-key-here
   ```
   
   Or edit `config.yaml` directly:
   ```yaml
   api_keys:
     openai_api_key: "your-openai-key-here"
     anthropic_api_key: "your-anthropic-key-here"
   ```

3. **Tune Silence Detection**:
   ```bash
   python main.py --tune
   ```
   Follow the prompts to calibrate the silence threshold for your microphone and environment.

4. **Configure Hyprland**:
   Add to your `~/.config/hypr/hyprland.conf`:
   ```
   # Default/General STT
   bind = ALT SHIFT, R, exec, cd /path/to/stt && python main.py
   
   # Context-specific shortcuts
   bind = ALT SHIFT, S, exec, cd /path/to/stt && python main.py --profile slack
   bind = ALT SHIFT, E, exec, cd /path/to/stt && python main.py --profile email
   bind = ALT SHIFT, T, exec, cd /path/to/stt && python main.py --profile todo
   bind = ALT SHIFT, O, exec, cd /path/to/stt && python main.py --profile obsidian
   ```

## Usage

### Basic Usage
```bash
python main.py
```
The system will **immediately** start recording while loading the Whisper model in the background. You can start speaking right away! When you stop speaking (silence detected), the system waits for model loading to complete (if needed) and then transcribes your audio.

### Using Different Profiles
```bash
python main.py --profile slack       # Format for Slack messages
python main.py --profile email       # Format as professional email
python main.py --profile todo        # Create actionable todo items
python main.py --profile obsidian    # Format as Obsidian markdown note
python main.py --profile code_comment # Format as code documentation
python main.py --profile meeting_notes # Structure as meeting notes
```

### List Available Profiles
```bash
python main.py --list-profiles
```

### Tuning Mode
```bash
python main.py --tune
```
Calibrates the silence detection threshold for your specific setup.

### Verbose Logging
```bash
python main.py --verbose
```
Enables detailed debug logging.

## Configuration

Edit `config.yaml` to customize:

```yaml
audio:
  silence_threshold: 30        # Volume threshold for silence detection
  silence_duration: 2.0        # Seconds of silence before stopping
  max_recording_time: 30.0     # Maximum recording time (safety)

whisper:
  model: base                  # tiny, base, small, medium, large
  language: auto               # Language code or 'auto'
  device: cuda                 # cuda, cpu, or auto
  compute_type: float16        # float16, int8, or float32
  load_timeout: 60             # seconds to wait for model loading

llm:
  provider: openai             # openai or anthropic
  model: gpt-3.5-turbo        # Model to use for text refinement
  
clipboard:
  auto_paste: false            # Automatically paste to active window
```

## Logging

The system logs to both console and `stt.log` file:
- INFO: General operation messages
- DEBUG: Detailed debugging information (use `--verbose`)
- ERROR: Error conditions
- WARNING: Important notices

## Troubleshooting

- **Audio Issues**: Make sure your microphone is working and accessible
- **Silence Detection**: Run `python main.py --tune` to recalibrate
- **API Errors**: Verify your API keys are correct and have sufficient credits
- **Auto-paste Issues**: Ensure `xdotool` is installed and accessible

## File Structure

- `main.py`: Main entry point with argument parsing
- `config.py`: YAML configuration management
- `audio_recorder.py`: Microphone recording with silence detection
- `stt_processor.py`: Whisper integration
- `llm_refiner.py`: LLM text refinement
- `clipboard_manager.py`: Clipboard and auto-paste features
- `config.yaml`: User configuration file
- `stt.log`: Application log file