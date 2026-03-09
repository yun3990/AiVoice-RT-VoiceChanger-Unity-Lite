# AiVoice Realtime Voice Changer

Free evaluation build for testing only (non-commercial, no redistribution). Full Unity SDK integration is available in the Pro version.

Real-time RVC-based voice conversion app for Windows.  
Runs a local Python WebSocket server and converts your voice through an RVC model with low latency.

> Latency varies depending on your GPU. RTX 3060+ recommended for best results. Lower GPUs may work with reduced settings.

---

## Download

**[→ Download Latest Release](https://github.com/yun3990/AiVoice-RT-VoiceChanger-Unity-Lite/releases)**

> **Want Unity SDK integration?**  
> The Pro version includes full C# source, sample scene, and Unity SDK.  
> **Available on Booth** *(coming soon)*

---

## Requirements

| | |
|---|---|
| OS | Windows 10/11 64-bit |
| GPU | NVIDIA GPU (CUDA-capable, driver 551.61+ recommended / 528.33+ minimum) |
| RAM | 8GB+ |
| Storage | ~6GB (after setup) |
| Internet | Required for first-time setup only |

> **AMD GPU / CPU-only is not supported.**  
> **Mac and Linux are not currently supported.** Windows only.  
> If setup fails due to a driver or CUDA issue, please check that your NVIDIA driver is up to date before reporting.

---

## First-Time Setup

1. Download and extract the zip from [Releases](https://github.com/yun3990/AiVoice-RT-VoiceChanger-Unity-Lite/releases)
2. Place base model files in `ServerPack/models/base/` (see below)
3. Run `AiVoice.exe`

Setup runs automatically on first launch.  
If the server does not start automatically, run `ServerPack/launch/start_server.bat` manually.  
Typical duration: **10–20 minutes** (may be longer depending on internet speed and hardware).

The following will be downloaded and installed automatically:
- Python 3.10.11 (embedded, isolated — no existing Python installation required, no conflicts)
- PyTorch 2.5.1 + CUDA 12.4 (~3GB)
- Required Python libraries

---

## Base Models (Required)

Base model files are provided as a **separate download** via the [Releases](https://github.com/yun3990/AiVoice-RT-VoiceChanger-Unity-Lite/releases) page.

1. Download the base models zip from [Releases](https://github.com/yun3990/AiVoice-RT-VoiceChanger-Unity-Lite/releases)
2. Extract and place files here:

```
ServerPack/
└── models/
    └── base/
        ├── hubert_base.pt
        └── rmvpe.onnx
```

> **The server will not start without these files.**

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

## File Structure

```
AiVoice-RT-VoiceChanger-Unity-Lite/
├── AiVoiceRelease/
│   └── AiVoice.exe              ← run this
├── ServerPack/
│   ├── launch/
│   │   ├── start_server.bat
│   │   └── setup_env.bat
│   ├── config/
│   │   └── server_config.json
│   ├── core/
│   │   └── rvc_ws_server.py
│   ├── models/
│   │   ├── base/            ← hubert_base.pt, rmvpe.onnx (separate download)
│   │   └── voices/          ← your RVC models here
│   ├── wheels/
│   └── OPEN_SOURCE_LICENSES.txt
├── ThirdParty/
│   └── wokada/
├── README.md
└── README_Lite.txt
```

---

## Pro Version (Unity SDK)

The Pro version includes everything in this release plus:

- Full Unity C# source (`ServerLauncher.cs`, `RvcControlPanel.cs`, etc.)
- Sample scene — ready to test out of the box
- Unity project integration support

**Available on Booth** *(coming soon)*

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Server does not start | Check that base models are in `ServerPack/models/base/` |
| Setup hangs or fails | Check your internet connection and NVIDIA driver version |
| No voice output | Make sure your mic is set as the default input device in Windows |
| Model not appearing | Click Refresh in the control panel, or check folder structure |
| High latency / stuttering | Use a smaller BlockFrames value or try the Low Latency preset |
| RVC v2 .pth error | v2 .pth is not supported — export to ONNX first |

---

## License (Lite / Evaluation Build)

Copyright © 2025 yun3990. All rights reserved.

- This Lite build is provided for evaluation purposes only.
- Personal, non-commercial use is permitted for evaluation.
- Redistribution, reuploading, repackaging, and modification for distribution are prohibited.
- Commercial use and redistribution rights are granted only with the paid version (Booth).

> Third-party libraries included in this package are governed by their respective licenses.  
> See [`ServerPack/OPEN_SOURCE_LICENSES.txt`](ServerPack/OPEN_SOURCE_LICENSES.txt) for details.

---

## Links

- **GitHub:** https://github.com/yun3990/AiVoice-RT-VoiceChanger-Unity-Lite
- **Discord:** https://discord.gg/dNuxwnyszS
- **Booth (Pro):** https://booth.pm/ko/items/8065372
- **Issues / Bug Reports:** https://github.com/yun3990/AiVoice-RT-VoiceChanger-Unity-Lite/issues
