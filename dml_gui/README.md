a simple gui wrapper for onnxruntime ep DirectMl on windows platform
> build in py=3.9, msvc is needed to build pyopenjtalk
> if use --windowed, need to download [open_jtalk_dic_utf_8-1.11](https://github.com/r9y9/open_jtalk/releases/download/v1.11.1/open_jtalk_dic_utf_8-1.11.tar.gz), decompress it and put it in the pyopenjtalk folder in exe dir 

```pwsh
set PYTHONOPTIMIZE=2
pyinstaller.exe `
 --path C:\Users\ccds\Desktop\gitrepo\vits_web_demo\dml\app `
 .\main.py -n vits_directml `
--windowed
```