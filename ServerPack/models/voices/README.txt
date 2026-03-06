==============================================
 AiVoice - Voice Model Folder
==============================================

Place your RVC voice models in this folder.

Folder structure:
  voices/
    your_model_name/
      your_model_name.onnx  (or .pth)
      meta.json             (optional)

meta.json format:
  {
    "label": "Display Name",
    "f0_up_key": 0
  }

Example:
  voices/
    my_voice/
      my_voice.onnx
      meta.json

Notes:
- Each model must be in its own subfolder.
- Supported formats: .onnx, .pth
- meta.json is optional but recommended for display name and pitch settings.

----------------------------------------------
 [한국어]
----------------------------------------------

이 폴더에 RVC 보이스 모델을 배치하세요.

폴더 구조:
  voices/
    모델이름/
      모델이름.onnx  (또는 .pth)
      meta.json      (선택)

meta.json 형식:
  {
    "label": "표시 이름",
    "f0_up_key": 0
  }

예시:
  voices/
    my_voice/
      my_voice.onnx
      meta.json

주의사항:
- 각 모델은 반드시 개별 하위 폴더 안에 있어야 합니다.
- 지원 형식: .onnx, .pth
- meta.json은 선택사항이지만 표시 이름과 피치 설정을 위해 권장합니다.
