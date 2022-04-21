#!/bin/bash

docker system prune -a -f
rm -r submission
reprounzip docker setup submission.rpz submission
reprounzip docker run submission
