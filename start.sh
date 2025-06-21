#!/bin/bash
# start.sh
cd scripts
gunicorn main:app --bind=0.0.0.0:$PORT