@echo off
title Neo4j - graph
podman exec -it flashsale-neo4j cypher-shell
if errorlevel 1 pause
