@echo off
rem Windows installer entry point: install.bat [auto|cpu|cuda]
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
