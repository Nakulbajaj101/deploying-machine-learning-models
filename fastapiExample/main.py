from fastapi import FastAPI


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/square")
async def square(num: int) -> dict:
    result = num ** 2
    return {"squared": result}

@app.get("/getData")
async def data(val: str) -> dict:
    if val in ["new", "old"]:
        result = val
        return {"data": result}
    raise ValueError("Invalid val, Acceptable inputs 'old' or 'new'")
