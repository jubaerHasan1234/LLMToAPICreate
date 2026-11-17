import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel
from fastapi import  APIRouter

router = APIRouter()



# text --> input token ---> model ---> output token --> text

tokenizer= AutoTokenizer.from_pretrained("AiModel/llama")
model = AutoModelForCausalLM.from_pretrained(
    "AiModel/llama",
    dtype=torch.bfloat16,
    device_map="auto"
)

# request schema
class ChatRequest(BaseModel):
    prompt: str

# api function

@router.post("/chat-llama")
def chat(request: ChatRequest):

          # Process Prompt
        messages=[{"role": "user", "content": request.prompt}]

        # input messages ---> input token
        InputToken = tokenizer.apply_chat_template(
            messages,
            add_special_tokens=True,
            tokenizer=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device)

        #Input token ---> Model--> Output Token
        OutputToken = model.generate(**InputToken,max_new_tokens=128)

        #Output Token ---> Output Text
        result = tokenizer.decode(OutputToken[0][InputToken['input_ids'].shape[-1]:], skip_special_tokens=True)

        return {"content": result}