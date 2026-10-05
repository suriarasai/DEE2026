@echo off
title MongoDB - documents
podman exec -it flashsale-mongo mongosh flashsale
if errorlevel 1 pause
