@echo off
echo DOWNLOADS_DIR=%USERPROFILE%\Downloads> .env
echo PROJECT_DIR=%CD%>> .env
echo .env created:
type .env
docker compose up --build