from fastapi import FastAPI
from pydantic import BaseModel

from my_agent_graph import app_graph

app = FastAPI()

class Request(BaseModel):
    input: str

@app.post("/invoke")
async def invoke(req: Request):
    result = app_graph.invoke({
        "input": req.input
    }) # type: ignore

    return result
