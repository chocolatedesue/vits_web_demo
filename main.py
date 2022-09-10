
from io import BytesIO
import librosa

from soundfile import write
from fastapi import FastAPI, status,Form,File,Response
from fastapi.responses import JSONResponse
from urllib.parse import unquote


from tts_model_wrapper import raw_vits
from myutil.find_file import find_by_postfix
from myutil.data_entity import Text,vc_data,tts_data
from text import _clean_text


SUPPORTED_CLEANNER = ["japanese_cleaners","japanese_cleaners2"]
device = 'cpu'

app = FastAPI()

vits_model_0 = raw_vits(find_by_postfix("/mydata","json"),find_by_postfix("/mydata","pth"))
VITS_MODEL_LIST = [vits_model_0]


@app.get("/api/util/clean")
async def clean_text(text:Text):
    text,cleanner = text.text,text.cleanner
    if not cleanner:
        return text
    elif cleanner not in SUPPORTED_CLEANNER:
        meg = { "detail" : [{"meg":"error: unsupported cleanner"}]}
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content=meg) 
    else:
        return _clean_text(text,[cleanner])


@app.get("/api/vits/tts")
async def synthesis (_:tts_data):
    text,model_id,iscleaned,sid = _.text,_.model_id,_.is_cleaned,_.sid
    text = unquote(text)
    model:raw_vits = VITS_MODEL_LIST[model_id] 
    if iscleaned:
        cleanned_text_tensor = model.get_text(text,cleaned=True)
    else:
        cleanned_text_tensor = model.get_text(text)
    audio = model.tts(cleanned_text_tensor,sid)

    with BytesIO() as audio_data:
        write(audio_data, audio, model.hps_ms.data.sampling_rate,format='wav')
        response = Response(audio_data.getvalue(),status_code=200,media_type="audio/wav") 
        return response
 
@app.post("/api/vits/vc")
async def voice_converse(src_audio:bytes=File(),src_id:int=Form(...),target_id:int=Form(...),model_id:int=Form(0)):
    model:raw_vits = VITS_MODEL_LIST[model_id]
    bt = BytesIO(src_audio)
    tmp = librosa.load(bt,sr=model.hps_ms.data.sampling_rate)[0]
    audio = model.vc(src_id,target_id,tmp)
    with BytesIO() as audio_data:
        write(audio_data, audio, model.hps_ms.data.sampling_rate,format='wav')
        response = Response(audio_data.getvalue(),status_code=200,media_type="audio/wav") 
        return response            






if __name__=='__main__':
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8088, log_level="warning")

