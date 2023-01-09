mkdir -p model

python vits/export_onnx.py --checkpoint model/g_940_demo.pth --cfg model/config.json \
    --onnx_model model/res.onnx 