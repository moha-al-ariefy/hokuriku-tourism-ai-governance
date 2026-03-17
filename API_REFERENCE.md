# API Reference

## src/privacy_nlp.py

### get_nlp_model()
Lazy-load spaCy Japanese NER model (`ja_core_news_sm`) when installed.

### sanitize_text(text: str) -> str
Redacts accidental PII in free text:
- email addresses
- phone numbers
- PERSON entities (spaCy)

Returns text with redaction tokens such as `[REDACTED_EMAIL]`.

### apply_privacy_layer(df: pd.DataFrame, text_columns: List[str]) -> pd.DataFrame
Applies `sanitize_text` to the specified DataFrame columns and returns a copied sanitized frame.

## src/kansei.py

### run_zero_shot_diagnostics(
- `survey_df: pd.DataFrame`
- `reporter: Reporter | None = None`
- `max_samples: int | None = 3000`
- `text_max_chars: int = 512`
) -> dict[str, float]

Runs zero-shot classification on detractor free text and returns percentage distribution by category.

Current labels:
- `weather conditions`
- `poor transportation`
- `language barrier`
- `lack of information`
- `pricing`

## src/visualizer.py

### plot_opportunity_gap_drivers(
- `driver_percentages: dict[str, float]`
- `out_path: str`
- `reporter: Reporter`
- `dpi: int = 300`
) -> `matplotlib.figure.Figure | None`

Plots complaint-driver percentages and writes EN/JA PNG outputs via `Reporter`.
