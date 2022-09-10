import os
def find_by_postfix(dir_path:str,postfix:str):
    for i in os.listdir(dir_path):
        res = i.split('.')[-1]
        if res==postfix:
            return os.path.join(dir_path,i)
        
    raise FileNotFoundError(f"Cann't find file endwith {postfix}, please check dir path")   