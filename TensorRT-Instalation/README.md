# TesnorRT Instalation

Source: https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html#installing-tar

برای cuda 11.2:

1- دانلود tensorrt 8.4.1.5 از این لینک:

https://developer.nvidia.com/tensorrt

2- اجرای کامند های زیر:

```
tar -xzvf TensorRT-8.4.1.5.Linux.x86_64-gnu.cuda-11.6.cudnn8.4.tar.gz

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:TensorRT-8.4.1.5/lib

cd TensorRT-8.4.1.5/python/
python3 -m pip install tensorrt-8.4.1.5-cp37-none-linux_x86_64.whl

cd TensorRT-8.4.1.5/uff/
python3 -m pip install uff-0.6.9-py2.py3-none-any.whl

# Check the installation with:
 which convert-to-uff
 
cd TensorRT-8.4.1.5/graphsurgeon/
python3 -m pip install graphsurgeon-0.4.6-py2.py3-none-any.whl

cd TensorRT-8.4.1.5/onnx_graphsurgeon/
python3 -m pip install onnx_graphsurgeon-0.3.12-py2.py3-none-any.whl

```

