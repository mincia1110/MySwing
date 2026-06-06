#!/bin/bash
set -e
export PATH="$HOME/.local/bin:$PATH"
cd /home/kim_minchul/workspace/MySwing/frontend
npm run test:run -- --reporter=default
