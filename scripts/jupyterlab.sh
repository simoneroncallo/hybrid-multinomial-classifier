#!/bin/bash
TMPDIR=$(mktemp -d)
rsync -avc --exclude-from=./rsyncignore "$PWD/" "$TMPDIR"
find "$TMPDIR/" -type d -exec chmod 777 {} \;
find "$TMPDIR/" -type f -exec chmod 666 {} \;
echo "Mounted at $TMPDIR"

echo "Starting Docker..."
docker run --rm --cap-drop=ALL --security-opt=no-new-privileges:true \
	--user=jupyteruser -p 127.0.0.1:8888:8888 \
	--cpus=16 --memory=16384m \
	-v "$TMPDIR":/home/jupyteruser/work \
	multinomial-classifier:latest

rsync -avc --no-perms --exclude-from=./rsyncignore "$TMPDIR/" "$PWD"
rm -rf "$TMPDIR"

echo "Completed"
