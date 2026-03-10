# AiVoice — Real-time RVC Voice Changer

Real-time RVC-based voice conversion app for Windows.  
This Lite build is intended for testing a local runtime flow used in a Unity-related workflow.

Runs a local Python WebSocket server and converts your voice through an RVC model with low latency.

No cloud, no API key — runs entirely on your machine.  
Bring your own RVC voice models.

> Latency varies depending on your GPU. RTX 3060+ recommended for best results. Lower GPUs may work with reduced settings.

---

## Download

**[→ Download Latest Release](https://github.com/yun3990/AiVoice-RT-VoiceChanger-Unity-Lite/releases)**

---

## Requirements

| | |
|---|---|
| OS | Windows 10/11 64-bit |
| GPU | NVIDIA GPU with CUDA support |
| RAM | 8GB+ |
| Storage | ~6GB (after setup) |
| Internet | Required for first-time setup only |

> **AMD GPU / CPU-only is not supported.**  
> **Mac and Linux are not currently supported.** Windows only.  
> Lower-end NVIDIA GPUs may still run the app with reduced settings.  
> For example, **GTX 1660 Super worked in limited testing, but smooth real-time performance is not guaranteed.**  
> If setup fails, update your NVIDIA driver first.

---

## Getting Started

1. Download and extract the zip from [Releases](https://github.com/yun3990/AiVoice-RT-VoiceChanger-Unity-Lite/releases)
2. Place your RVC model in `ServerPack/models/voices/YourModelName/`
3. Run `AiVoice.exe`

Setup runs automatically on first launch.  
First-time setup may take a while depending on internet speed and hardware.  
If the server does not start automatically, run `ServerPack/launch/start_server.bat` manually.

The following will be downloaded and installed automatically:
- Python 3.10.11 (embedded, isolated — no conflicts with your system Python)
- PyTorch 2.5.1 + CUDA 12.4 (~3GB)
- Required Python libraries

---

## Base Models

Base model files (`hubert_base.pt`, `rmvpe.onnx`) are included in the package.  
They are located at:

```
ServerPack/
└── models/
    └── base/
        ├── hubert_base.pt
        └── rmvpe.onnx
```

> **The server will not start without these files.** Do not move or delete them.

---

## Adding Voice Models

Place your RVC model files inside:

```
ServerPack/
└── models/
    └── voices/
        └── YourModelName/
            └── your_model.onnx   ← or .pth
```

Supported formats:
- `.onnx` — RVC ONNX models
- `.pth` — RVC v1 (256-dim) only

> **RVC v2 `.pth` models (768-dim) are not supported.** Use an ONNX-exported version instead.

Optional: place `index.bin` or `added.index` in the same folder for FAISS index support.

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Server does not start | Check that base models are in `ServerPack/models/base/` |
| Setup hangs or fails | Check your internet connection and update your NVIDIA driver |
| No voice output | Make sure your mic is set as the default input device in Windows |
| Model not appearing | Click Refresh in the control panel, or check folder structure |
| High latency / stuttering | Use a smaller BlockFrames value or try the Low Latency preset |
| RVC v2 .pth error | v2 .pth is not supported — export to ONNX first |

---

## License

Copyright © 2025 yun3990. All rights reserved.

- This Lite build is provided for personal, non-commercial use only.
- Redistribution, reuploading, repackaging, and modification for distribution are prohibited.

> Third-party libraries included in this package are governed by their respective licenses.  
> See [`ServerPack/OPEN_SOURCE_LICENSES.txt`](ServerPack/OPEN_SOURCE_LICENSES.txt) for details.

---

## Links

- **GitHub:** https://github.com/yun3990/AiVoice-RT-VoiceChanger-Unity-Lite
- **Issues / Bug Reports:** https://github.com/yun3990/AiVoice-RT-VoiceChanger-Unity-Lite/issues
- **Discord:** https://discord.gg/dNuxwnyszS
