#! /bin/bash

rm -rf dist
uv build
git pull
sudo docker compose pull
# sudo cp -r ~/.cache/pip /root/.cache
# sudo docker compose build
sudo docker compose down
sudo docker compose up --remove-orphans -d
sudo docker compose logs -f
