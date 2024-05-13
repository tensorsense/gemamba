#!/bin/bash

cd /data/gemamba

# Check if the init script has been run before
if [ ! -f "/app/init_done" ]; then
    chmod +x .devcontainer/postCreateCommand.sh
    .devcontainer/postCreateCommand.sh
    # Mark initialization as done
    touch /init_done
fi

# Continue with the main command
exec "$@"
