import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from app.gemma_api import router as gemma_router
# from app.llama_api import router as llama_router
from app.whisper_api import router as whisper_router
app = FastAPI()

# Recommended cleaner way to add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
)

# app.include_router(gemma_router)
# app.include_router(llama_router)
app.include_router(whisper_router)
@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/ping")
async def ping():
    return {"ping": "pong"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)