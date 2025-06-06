#! /bin/bash

rm -rf dist
uv sync  --all-groups
uv export --no-editable --no-emit-project -o requirements.txt > /dev/null
uv build
git pull
sudo docker compose pull
sudo docker compose build
sudo docker compose down
sudo docker compose up --remove-orphans -d
sudo docker compose logs -f
