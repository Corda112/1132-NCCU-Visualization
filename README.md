# 1132-NCCU-Visualization
Information Visualization

## Development

Start the backend server:

```bash
cd backend && node server.js
```

First-time setup requires installing dependencies:

```bash
npm install --prefix backend
npm install --prefix frontend
```

An optional `setup.sh` script automates these steps and attempts to rebuild
`sqlite3` for the current environment:

```bash
./setup.sh
```

Start the React frontend:

```bash
cd frontend && npm start
```

Run tests:

```bash
npm test
```

### Troubleshooting

If the backend fails with `invalid ELF header`, rebuild `sqlite3` from source:

```bash
npm rebuild sqlite3 --build-from-source --prefix backend
```

Running `npm test` at the project root will execute the frontend test script.

When the Node `sqlite3` module cannot be installed, the server will attempt to
query `db.sqlite3` using the `sqlite3` command line tool. If that also fails,
the API uses bundled JSON files for sentiment and term data only, while the
financial endpoints will return empty results.
