#!/usr/bin/env bash
cmd="uvicorn my_agent_server:app --host 0.0.0.0 --port 8000"
echo $cmd
eval $cmd
