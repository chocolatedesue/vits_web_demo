from pydantic import BaseModel

class vc_data(BaseModel):
    src_audio:bytes
    src_id:int
    target_id:int
    model_id:int=0
    

class Text(BaseModel):
    text: str
    cleanner: str

class tts_data(BaseModel):
    text:str
    model_id:int=0
    sid:int=0
    is_cleaned:bool=False