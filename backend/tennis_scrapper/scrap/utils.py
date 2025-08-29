def find_div_by_text(divs, start_with: str):
    return next(
        filter(
            lambda d: d.get_text(strip=True).startswith(start_with),
            divs,
        ),
        None,
    )
