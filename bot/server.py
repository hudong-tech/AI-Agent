from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/chat")
def chat():
    return {"response": "I am a bot"}

@app.post("/add_ursl")
def add_ursl():
    return {"response": "URLs added successfully"}

@app.post("/add_pdfs")
def add_pdfs():
    return {"response": "PDFs added successfully"}

@app.post("/add_texts")
def add_texts():
    return {"response": "Texts added successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)