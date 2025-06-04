from onnxruntime.quantization import quantize_dynamic, QuantType
import onnx

# Quantize your models
def quantize_model(input_path, output_path):
    quantize_dynamic(
        input_path,
        output_path,
        weight_type=QuantType.QUInt8
    )
    
    # Check the size reduction
    original = onnx.load(input_path)
    quantized = onnx.load(output_path)
    print(f"Original size: {len(original.SerializeToString()) / 1024 / 1024:.1f} MB")
    print(f"Quantized size: {len(quantized.SerializeToString()) / 1024 / 1024:.1f} MB")

# Quantize both models
quantize_model("./onnx/emotion/model.onnx", "./onnx/emotion/model_quantized.onnx")
quantize_model("./onnx/mpnet/model.onnx", "./onnx/mpnet/model_quantized.onnx")