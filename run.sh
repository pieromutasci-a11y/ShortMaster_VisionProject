#!/bin/bash

# Consente al container di accedere all'interfaccia grafica (fondamentale per Gazebo e RViz)
xhost +

# Crea la cartella di lavoro sul tuo PC (l'opzione -p evita errori se la cartella esiste già)
mkdir -p $(pwd)/ros_ws

# Avvia il container mappando la cartella e rimuovendolo alla chiusura (--rm)
docker run -it --rm --name pal_docker \
               --device /dev/video0:/dev/video0 \
               --device /dev/dri:/dev/dri \
               -e DISPLAY=$DISPLAY \
               -v /tmp/.X11-unix:/tmp/.X11-unix \
               -v $(pwd)/ros_ws:/home/user/ros_workspace \
               -v $(pwd)/.bash_aliases:/home/user/.bash_aliases \
               palrobotics/public-simulation-humble-public bash



