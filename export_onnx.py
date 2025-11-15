import torch, timm
from pathlib import Path

ROOT = Path.home() / "Downloads" / "board_dataset_work"
ckpt_path = ROOT / "models" / "mobilenetv2_led_best.pth"
onnx_path = ROOT / "models" / "mobilenetv2_led.onnx"

ckpt = torch.load(ckpt_path, map_location="cpu")
class_names = ckpt["class_names"]
img_size = ckpt["img_size"]

model = timm.create_model("mobilenetv2_100", pretrained=False, num_classes=len(class_names))
model.load_state_dict(ckpt["model"])
model.eval()

dummy = torch.randn(1, 3, img_size, img_size)

torch.onnx.export(
    model, dummy, onnx_path,
    input_names=["input"], output_names=["logits"],
    opset_version=13,
    dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    export_params=True,
)
print("Exported:", onnx_path)

sidecar = onnx_path.with_suffix(onnx_path.suffix + ".data")
if sidecar.exists():
    import onnx
    print("Merging external data into a single ONNX file...")
    m = onnx.load(str(onnx_path), load_external_data=True)
    onnx.save_model(m, str(onnx_path), save_as_external_data=False)

    m2 = onnx.load(str(onnx_path), load_external_data=False)
    def model_uses_external(mm):
        from onnx import external_data_helper as edh
        return any(edh.uses_external_data(t) for t in mm.graph.initializer)
    if not model_uses_external(m2):
        try:
            sidecar.unlink()
        except Exception:
            pass
    print("Merged single-file ONNX:", onnx_path)
