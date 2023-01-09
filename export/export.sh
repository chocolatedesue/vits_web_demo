mkdir -p model

python vits/export_onnx.py --checkpoint model/19_1265_demo.pth --cfg model/config.json \
    --onnx_model model/res.onnx 