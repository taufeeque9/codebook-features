#!/bin/bash
set -ex
echo -n "Retrieving repositories..."
if [ ${GIT_UPSTREAM:-none} == "none" ]; then
	echo "Exiting, $GIT_UPSTREAM not set"
	exit
fi
if [ -d code ]; then
	echo -n "updating existing repository"
	pushd code
	git pull
	git submodule update --remote
else
	echo -n "cloning fresh repository"
	git clone $GIT_UPSTREAM code --depth 1 --branch ${GIT_BRANCH:-master}
	pushd code
	git submodule init
	git submodule update
fi
find -name "*requirements.txt" -exec pip install -r {} \;
find -name "setup.py" -exec bash -c 'cd $(dirname {}) && python $(basename {}) develop' \;
echo "Done"
