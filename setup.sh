#!/bin/sh
set -e
npm install --prefix backend
npm install --prefix frontend
npm rebuild sqlite3 --build-from-source --prefix backend || echo "sqlite3 rebuild failed - ensure build tools and network access"
