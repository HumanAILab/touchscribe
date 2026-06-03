# TouchScribe

> ⚠️ **Under Construction — Not Ready for Use**
>
> This repository is a work in progress. The code is incomplete, may be unstable,
> and is **not yet ready for production or general use**. APIs, structure, and
> functionality are subject to change without notice. Use at your own risk.

TouchScribe is a research system that combines real-time hand/gesture tracking,
object recognition, and multimodal language models to provide interactive audio
descriptions of the physical world. It streams camera frames, detects objects
and hand gestures, and uses LLM/TTS services to generate and speak captions.

## Features

- Real-time hand and gesture recognition
- Object detection and segmentation (YOLO / Grounded-SAM based)
- Color and shape recognition utilities
- Multimodal captioning via GPT-4V
- Text-to-speech playback (OpenAI TTS and Azure Speech)
- Firebase Realtime Database for client/server communication
- Vector memory store (LanceDB) for retrieval

## Requirements

- Python 3.10+
- See [`requirements.txt`](requirements.txt) for Python dependencies:

```bash
pip install -r requirements.txt
```

## Configuration

This project relies on third-party services. **No credentials are stored in the
repository** — provide your own via environment variables and local files.

| Variable | Purpose |
| --- | --- |
| `OPENAI_API_KEY` | OpenAI API key (GPT-4V, TTS) |
| `AZURE_SPEECH_KEY` | Azure Speech subscription key (optional TTS) |
| `AZURE_SPEECH_REGION` | Azure Speech region (defaults to `eastus`) |

```bash
export OPENAI_API_KEY="your-openai-key"
export AZURE_SPEECH_KEY="your-azure-key"
export AZURE_SPEECH_REGION="eastus"
```

### Firebase credentials

The app initializes the Firebase Admin SDK from a service-account JSON file
(e.g. `soundcaption-a6e7d-firebase-adminsdk-*.json`). This file is **ignored by
git** and must be supplied locally. Download your own service-account key from
the Firebase console and place it in the project root, keeping the filename
referenced by the code (or update the path in `main.py` / `remote_server.py`).

## Project layout

```
main.py                     # Main entry point (client/visual server)
remote_server.py            # Remote processing server
handscribe_remote_server.py # HandScribe remote server
memory_manager.py           # LanceDB-backed vector memory
utils/                      # Recognition, gesture, firebase, speech modules
masking/                    # Segmentation / Grounded-SAM utilities
config/                     # Server configuration
```

## Notes on large files

Model weights (`*.pt`, `*.pth`, `models/`), datasets (`image_data/`,
`handscribe_db/`, `*.lance`), media, and assets are excluded via `.gitignore`
to keep the repository small. Download or generate these locally as needed.

## License

Released under the [MIT License](LICENSE).
