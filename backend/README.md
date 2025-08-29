# Tennis Scrapper

An asynchronous and modular scrapper to collect tennis tournaments, matches, and players data from specialized websites.

## Features
- Asynchronous scraping of tournaments, matches, and players
- Player cache management with `contextvars` for async safety
- User-friendly CLI with [Typer](https://typer.tiangolo.com/)
- Data storage in SQLite using [SQLModel](https://sqlmodel.tiangolo.com/)
- Advanced logging with [Loguru](https://github.com/Delgan/loguru)
- Configuration management via TOML files

## Installation

1. **Clone the repository**
   ```bash
   git clone <repo-url>
   cd tennis_scrapper
   ```
2. **Install dependencies with [uv](https://github.com/astral-sh/uv)**
   ```bash
   uv sync
   ```

## Usage
For more details on the cli commands, you can refer to the dedicated [ documentation file](CLI_USAGE.md)
### Scrap tournaments
First, you have to collect tournament's data
```bash
uv run -m tennis_scrapper/main.py scrap-tournaments --from 2010 --to 2024
```

### Scrape matches
Then, you can collect matches data for the tournaments you scraped
```bash
uv run -m tennis_scrapper/main.py scrap-matches --from 2010 --to 2024
```



## Configuration
The `conf/settings.toml` file allows you to configure the base URL, scraping parameters, etc.

## Best practices
- Use of async for fast scraping
- Thread-safe cache to avoid redundant requests
- Clear separation of concerns (CLI, DB, scraping, utils)

## License
MIT

## Author
Pierre Carceller Meunier