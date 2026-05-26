#!/usr/bin/env bash
cmd="curl localhost:8000/invoke -X POST -H 'Content-Type: application/json' -d '{\"input\":\"hello\"}'"
echo $cmd
eval $cmd
