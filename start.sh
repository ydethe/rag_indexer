#! /bin/bash

uv export --no-editable --no-emit-project -o requirements.txt > /dev/null
rm -rf dist
uv build
sudo docker compose down
sudo docker compose up --remove-orphans --build -d
sudo docker compose logs -f
