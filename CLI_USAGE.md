# CLI

**Usage**:

```console
$ [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--install-completion`: Install completion for the current shell.
* `--show-completion`: Show completion for the current shell, to copy it or customize the installation.
* `--help`: Show this message and exit.

**Commands**:

* `scrap-tournaments`: Scrape tennis tournaments from the ATP and...
* `scrap-matches`: Scrape matches from tournaments between...

## `scrap-tournaments`

Scrape tennis tournaments from the ATP and WTA websites for a given year range. The scraped data is stored in the database.

**Usage**:

```console
$ scrap-tournaments [OPTIONS]
```

**Options**:

* `--from INTEGER`: Start year (default: 1990)  [default: 1990]
* `--to INTEGER`: End year (default: current year)  [default: 2025]
* `--help`: Show this message and exit.

## `scrap-matches`

Scrape matches from tournaments between specified years, optionally clearing Match and Player tables. This command fetches match data, player details, and stores them in the database.The database should contain Tournament data before running this command.

**Usage**:

```console
$ scrap-matches [OPTIONS]
```

**Options**:

* `--from INTEGER`: Start year (default: 1990)  [default: 1990]
* `--to INTEGER`: End year (default: current year)  [default: 2025]
* `--clear`: Clear Match and Player tables before scraping
* `--help`: Show this message and exit.

