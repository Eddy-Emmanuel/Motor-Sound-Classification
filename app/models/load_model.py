import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download, login

load_dotenv()

login(token=os.getenv("HF_TOKEN"))

local_dir = snapshot_download(
    repo_id="Eddy-Emmanuel/Motor_Sound_Classifier",
    allow_patterns="*.pt"
)
