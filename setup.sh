#!/bin/bash
echo "DOWNLOADS_DIR=$HOME/Downloads" > .env
echo "PROJECT_DIR=$(pwd)" >> .env
echo ".env created:"
cat .env
docker compose up --build