#!/bin/bash
set -e
ln -sf "$HOME/.local/node/bin/node" "$HOME/.local/bin/node"
ln -sf "$HOME/.local/node/bin/npm" "$HOME/.local/bin/npm"
ln -sf "$HOME/.local/node/bin/npx" "$HOME/.local/bin/npx"
export PATH="$HOME/.local/bin:$PATH"
node --version
npm --version
