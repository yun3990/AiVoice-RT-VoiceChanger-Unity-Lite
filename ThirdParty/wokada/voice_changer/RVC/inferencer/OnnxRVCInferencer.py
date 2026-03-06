import torch
import onnxruntime
from const import EnumInferenceTypes
from voice_changer.RVC.deviceManager.DeviceManager import DeviceManager
from voice_changer.RVC.inferencer.Inferencer import Inferencer
import numpy as np


class OnnxRVCInferencer(Inferencer):
    def loadModel(self, file: str, gpu: int, inferencerTypeVersion: str | None = None):
        self.setProps(EnumInferenceTypes.onnxRVC, file, True, gpu)
        (
            onnxProviders,
            onnxProviderOptions,
        ) = DeviceManager.get_instance().getOnnxExecutionProvider(gpu)

        onnx_session = onnxruntime.InferenceSession(
            file, providers=onnxProviders, provider_options=onnxProviderOptions
        )

        # check half-precision
        first_input_type = onnx_session.get_inputs()[0].type
        if first_input_type == "tensor(float)":
            self.isHalf = False
        else:
            self.isHalf = True

        self.model = onnx_session

        self.inferencerTypeVersion = inferencerTypeVersion

        return self

    def infer(
        self,
        feats: torch.Tensor,
        pitch_length: torch.Tensor,
        pitch: torch.Tensor,
        pitchf: torch.Tensor,
        sid: torch.Tensor,
        convert_length: int | None,
    ) -> torch.Tensor:
        if pitch is None or pitchf is None:
            raise RuntimeError("[Voice Changer] Pitch or Pitchf is not found.")

        if self.isHalf:
            audio1 = self.model.run(
                ["audio"],
                {
                    "feats": feats.cpu().numpy().astype(np.float16),
                    "p_len": pitch_length.cpu().numpy().astype(np.int64),
                    "pitch": pitch.cpu().numpy().astype(np.int64),
                    "pitchf": pitchf.cpu().numpy().astype(np.float32),
                    "sid": sid.cpu().numpy().astype(np.int64)
                },
            )
        else:
            audio1 = self.model.run(
                ["audio"],
                {
                    "feats": feats.cpu().numpy().astype(np.float32),
                    "p_len": pitch_length.cpu().numpy().astype(np.int64),
                    "pitch": pitch.cpu().numpy().astype(np.int64),
                    "pitchf": pitchf.cpu().numpy().astype(np.float32),
                    "sid": sid.cpu().numpy().astype(np.int64)
                },
            )

        # --- audio output normalize (ALWAYS) ---
        out = audio1[0] if isinstance(audio1, (list, tuple)) else audio1
        y = np.asarray(out)

        y = np.squeeze(y)
        if y.ndim != 1:
            y = y.reshape(-1)

        y = y.astype(np.float32, copy=False)

        absmax = float(np.max(np.abs(y)) + 1e-12)
        if absmax > 1.2:
            if absmax > 100.0:
                y = y / 32768.0
            else:
                y = y / absmax

        y = np.clip(y, -1.0, 1.0).astype(np.float32, copy=False)
        return torch.from_numpy(y)


    def getInferencerInfo(self):
        inferencer = super().getInferencerInfo()
        inferencer["onnxExecutionProvider"] = self.model.get_providers()
        return inferencer