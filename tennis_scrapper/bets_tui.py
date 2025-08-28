#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from pathlib import Path
import pandas as pd
import typer

from rich.text import Text
from rich.panel import Panel

from textual.app import App, ComposeResult
from textual import events
from textual.containers import Horizontal
from textual.widgets import Header, Footer, Input, Static, DataTable, Label

from tennis_scrapper.db.db_utils import get_player_by_id
from tennis_scrapper.db.models import Match


app_cli = typer.Typer(add_completion=False)


COLUMNS_DEFAULT_ORDER = [
    "Date",
    "Round",
    "Players",
    "Predicted Winner",
    "Odds",
    "Model p(win)",
    "Implied p",
    "Edge %",
    "Kelly %",
    "Stake %",
    "Stake €",
    "EV %",
    "Decision",
]


class SummaryBar(Static):
    def update_summary(self, df: pd.DataFrame) -> None:
        if df is None or df.empty:
            self.update(Panel("No rows", border_style="cyan", title="Summary"))
            return

        if "Decision" in df.columns:
            bets = df[df["Decision"].str.startswith("💰")]
        else:
            bets = df

        total_stake = float(bets["Stake €"].sum()) if "Stake €" in bets.columns else 0.0

        exp_profit = 0.0
        if "EV %" in bets.columns and "Stake €" in bets.columns:
            for _, r in bets.iterrows():
                ev_pct = r.get("EV %")
                stake_eur = r.get("Stake €")
                if pd.notna(ev_pct) and pd.notna(stake_eur):
                    exp_profit += float(stake_eur) * (float(ev_pct) / 100.0)

        roi = (exp_profit / total_stake * 100.0) if total_stake > 1e-12 else 0.0

        text = Text.assemble(
            ("💸 Total stake: ", "bold"),
            (f"{total_stake:.2f} €   ", "bold green"),
            ("📈 Expected profit: ", "bold"),
            (f"{exp_profit:.2f} €   ", "bold green" if exp_profit >= 0 else "bold red"),
            ("📊 Expected ROI: ", "bold"),
            (f"{roi:.2f}%", "bold green" if roi >= 0 else "bold red"),
        )
        self.update(Panel(text, border_style="cyan", title="Summary"))


class BetsApp(App):
    CSS = """
    Screen {
        layout: vertical;
    }
    #topbar {
        height: 3;
    }
    #filter_label {
        width: 12;
        content-align: right middle;
        color: cyan;
    }
    #filter_input {
        content-align: left middle;
    }
    #status {
        height: 3;
        color: green;
    }
    #summary {
        height: 3;
    }
    DataTable {
        height: 1fr;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("f", "focus_filter", "Filter"),
        ("enter", "apply_filter", "Apply Filter"),
        ("s", "sort_current", "Sort Current Col"),
        ("d", "sort_date", "Sort by Date"),
        ("b", "toggle_bet_only", "BET only"),
        ("r", "reset_all", "Reset"),
    ]

    def __init__(self, df: pd.DataFrame, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.df_full = df.copy()
        cols = [c for c in COLUMNS_DEFAULT_ORDER if c in self.df_full.columns]
        extra = [c for c in self.df_full.columns if c not in cols]
        self.columns = cols + extra

        if "Date" in self.df_full.columns:
            self.df_full = self.df_full.sort_values("Date", ascending=True).reset_index(
                drop=True
            )
            self.current_sort_col = "Date"
            self.current_sort_asc = True
        else:
            self.current_sort_col = None
            self.current_sort_asc = True

        self.df_visible = self.df_full.copy()
        self.bet_only = False

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="topbar"):
            yield Label("🔎 Filter:", id="filter_label")
            yield Input(placeholder="Type player name...", id="filter_input")
        yield Static("", id="status")
        yield DataTable(id="table", zebra_stripes=True)
        yield SummaryBar(id="summary")
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one("#table", DataTable)
        for col in self.columns:
            table.add_column(col)
        self._refresh_table()
        self._refresh_summary()
        self._set_status()
        # Les handlers d'événements de Textual (on_input_changed / on_input_submitted)
        # gèrent désormais le filtrage. Aucune assignation directe sur le widget n'est nécessaire.

    # Handlers Textual pour le champ de recherche
    def on_input_changed(self, event: Input.Changed) -> None:
        if event.input.id == "filter_input":
            self._apply_filter(event.value)

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "filter_input":
            self._apply_filter(event.value)

    def on_key(self, event: events.Key) -> None:
        # ESC dans la barre de recherche -> reset input + table et focus sur la table
        if event.key == "escape" and isinstance(self.focused, Input):
            focused = self.focused
            if getattr(focused, "id", None) == "filter_input":
                focused.value = ""
                # Reset complet pour revenir à l'état initial
                self.action_reset_all()
                # Focus sur la table
                self.set_focus(self.query_one("#table", DataTable))
                event.stop()

    # Tri lorsque l'utilisateur clique sur l'en-tête d'une colonne du DataTable
    def on_data_table_header_selected(self, event: DataTable.HeaderSelected) -> None:
        # event.column_key peut être un index ou une clé; essayer de résoudre le nom de colonne
        col_key = getattr(event, "column_key", None)
        col_index = getattr(event, "column_index", None)
        col_name = None
        if col_key is not None and isinstance(col_key, str) and col_key in self.columns:
            col_name = col_key
        elif (
            col_index is not None
            and isinstance(col_index, int)
            and col_index < len(self.columns)
        ):
            col_name = self.columns[col_index]

        if not col_name:
            return

        # Basculer l'ordre si même colonne, sinon définir asc par défaut
        if self.current_sort_col == col_name:
            self.current_sort_asc = not self.current_sort_asc
        else:
            self.current_sort_col = col_name
            self.current_sort_asc = True

        q = self.query_one("#filter_input", Input).value
        self._apply_filter(q)

    def on_filter_input_changed(self, value: str) -> None:
        self._apply_filter(value)

    def _apply_filter(self, query: str) -> None:
        q = query.strip().lower()
        df = self.df_full
        if q:
            fields = [c for c in ["Players", "Predicted Winner"] if c in df.columns]
            if fields:
                mask = False
                for f in fields:
                    mask = (
                        (mask | df[f].astype(str).str.lower().str.contains(q, na=False))
                        if isinstance(mask, pd.Series)
                        else df[f].astype(str).str.lower().str.contains(q, na=False)
                    )
                df = df[mask]
        if self.bet_only and "Decision" in df.columns:
            df = df[df["Decision"].astype(str).str.startswith("💰")]
        if self.current_sort_col and self.current_sort_col in df.columns:
            df = df.sort_values(
                self.current_sort_col, ascending=self.current_sort_asc, kind="mergesort"
            )
        self.df_visible = df
        self._refresh_table()
        self._refresh_summary()
        self._set_status(query=query)

    def _refresh_table(self) -> None:
        table = self.query_one("#table", DataTable)
        table.clear(columns=False)

        for _, r in self.df_visible.iterrows():
            row = []
            is_bet = str(r.get("Decision", "")).startswith("💰")

            for col in self.columns:
                val = r.get(col, None)
                if pd.isna(val):
                    cell = "-"
                elif col == "Date" and hasattr(val, "date"):
                    cell = val.date().isoformat()
                else:
                    cell = str(val)

                # style only first col with color
                if col == self.columns[0]:
                    style = "green" if is_bet else "bright_black"
                    row.append(Text(cell, style=style))
                else:
                    row.append(cell)

            table.add_row(*row)

    def _refresh_summary(self) -> None:
        summary = self.query_one("#summary", SummaryBar)
        summary.update_summary(self.df_visible)

    def _set_status(self, query: str = "") -> None:
        status = self.query_one("#status", Static)
        parts = []
        parts.append(f"Rows: {len(self.df_visible)}/{len(self.df_full)}")
        if self.bet_only:
            parts.append("Filter: BET-only ✅")
        if query:
            parts.append(f"Search: '{query}'")
        if self.current_sort_col:
            arrow = "↑" if self.current_sort_asc else "↓"
            parts.append(f"Sort: {self.current_sort_col} {arrow}")
        status.update("   ".join(parts))

    # Actions
    def action_focus_filter(self) -> None:
        self.set_focus(self.query_one("#filter_input", Input))

    def action_apply_filter(self) -> None:
        q = self.query_one("#filter_input", Input).value
        self._apply_filter(q)

    def action_sort_current(self) -> None:
        if not self.columns:
            return
        col = self.columns[0]
        if self.current_sort_col == col:
            self.current_sort_asc = not self.current_sort_asc
        else:
            self.current_sort_col = col
            self.current_sort_asc = True
        q = self.query_one("#filter_input", Input).value
        self._apply_filter(q)

    def action_sort_date(self) -> None:
        if "Date" in self.df_full.columns:
            self.current_sort_col = "Date"
            self.current_sort_asc = True
            q = self.query_one("#filter_input", Input).value
            self._apply_filter(q)

    def action_toggle_bet_only(self) -> None:
        self.bet_only = not self.bet_only
        q = self.query_one("#filter_input", Input).value
        self._apply_filter(q)

    def action_reset_all(self) -> None:
        self.bet_only = False
        self.current_sort_col = "Date" if "Date" in self.df_full.columns else None
        self.current_sort_asc = True
        filter_input = self.query_one("#filter_input", Input)
        filter_input.value = ""
        self._apply_filter("")


def kelly_criterion(p: float, odds: float) -> float:
    if not (0.0 <= p <= 1.0) or odds <= 1.0:
        return 0.0
    b = odds - 1.0
    return max(0.0, (p * b - (1 - p)) / b)


def compute_bets(
    matches, predictions: pd.DataFrame, max_bet_fraction: float, bankroll: float
) -> pd.DataFrame:
    rows = []
    for match, (pred_class, pred_proba) in zip(matches, predictions.values):
        p1_player = get_player_by_id(match.player_1_id)
        p2_player = get_player_by_id(match.player_2_id)

        if pred_class == 0:
            predicted_player = p1_player
            p_win = 1.0 - pred_proba
            odd = match.player_1_odds
            side = "P1"
        elif pred_class == 1:
            predicted_player = p2_player
            p_win = float(pred_proba)
            odd = match.player_2_odds
            side = "P2"
        else:
            raise ValueError(f"Invalid predicted class: {pred_class}")

        valid_odd = odd and odd > 1.0
        implied = (1.0 / odd) if valid_odd else math.nan
        edge = (p_win - implied) if valid_odd else math.nan
        kelly_frac = kelly_criterion(p_win, odd) if valid_odd else 0.0

        stake_frac = min(kelly_frac, max_bet_fraction)
        stake_eur = bankroll * stake_frac
        ev = (p_win * odd - 1.0) if valid_odd else math.nan

        decision = "💰 BET" if kelly_frac > 0 else "⏸️ PASS"

        rows.append(
            {
                "Date": match.date,
                "Players": f"{p1_player.name} vs {p2_player.name}",
                "Predicted Winner": f"{predicted_player.name}({side})",
                "Odds": odd if valid_odd else None,
                "Model p(win)": round(p_win, 3),
                "Implied p": round(implied, 3) if valid_odd else None,
                "Edge %": round(100 * edge, 1) if valid_odd else None,
                "Kelly %": round(100 * kelly_frac, 2),
                "Stake %": round(100 * stake_frac, 2),
                "Stake €": round(stake_eur, 2),
                "EV %": round(100 * ev, 2) if valid_odd else None,
                "Decision": decision,
            }
        )

    df = pd.DataFrame(rows)
    return df


@app_cli.command()
def run(csv: Path, bankroll: float = 1000, max_bet_fraction: float = 0.05):
    """Run the interactive TUI with a CSV file of matches + probas."""

    # 1. Charger ton CSV brut
    raw = pd.read_csv(csv)

    predictions = raw[["predicted_class", "predicted_proba"]]

    matches = []
    for match_id in raw["match_id"]:
        json_file = csv.parent / "matches" / f"{match_id}.json"
        with open(json_file, "r") as f:
            match_data = json.load(f)
            match_data.pop("match_id", None)

        match = Match.model_validate(match_data)
        matches.append(match)

    # 3. Calculer le tableau enrichi
    df = compute_bets(matches, predictions, max_bet_fraction, bankroll)

    # 4. Lancer l’app
    app = BetsApp(df)
    app.run()


def main():
    app_cli()


if __name__ == "__main__":
    main()
