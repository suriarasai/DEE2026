@echo off
title Postgres - relational schema
podman exec -it flashsale-postgres psql -P pager=off -U postgres -d flashsale
if errorlevel 1 pause
