"""Output formatting utilities for CLI."""
import click


def print_header(text: str) -> None:
    """Print a formatted header."""
    click.echo()
    click.echo(click.style("=" * 60, fg='cyan'))
    click.echo(click.style(f"  {text}", fg='cyan', bold=True))
    click.echo(click.style("=" * 60, fg='cyan'))
    click.echo()


def print_success(text: str) -> None:
    """Print a success message."""
    click.echo(click.style(f"✓ {text}", fg='green', bold=True))


def print_error(text: str) -> None:
    """Print an error message."""
    click.echo(click.style(f"✗ {text}", fg='red', bold=True), err=True)


def print_warning(text: str) -> None:
    """Print a warning message."""
    click.echo(click.style(f"⚠ {text}", fg='yellow'))


def print_info(text: str) -> None:
    """Print an info message."""
    click.echo(text)


def print_progress(text: str) -> None:
    """Print a progress message."""
    click.echo(click.style(f"⟳ {text}", fg='blue'))


def print_table(headers: list, rows: list) -> None:
    """
    Print a simple table.
    
    Args:
        headers: List of column headers
        rows: List of row data (each row is a list)
    """
    # Calculate column widths
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            col_widths[i] = max(col_widths[i], len(str(cell)))
    
    # Print header
    header_row = " | ".join(
        h.ljust(w) for h, w in zip(headers, col_widths)
    )
    click.echo(click.style(header_row, bold=True))
    click.echo("-" * len(header_row))
    
    # Print rows
    for row in rows:
        row_str = " | ".join(
            str(cell).ljust(w) for cell, w in zip(row, col_widths)
        )
        click.echo(row_str)


def print_json(data: dict, indent: int = 2) -> None:
    """Print formatted JSON."""
    import json
    click.echo(json.dumps(data, indent=indent))


def confirm(text: str, default: bool = False) -> bool:
    """
    Ask for user confirmation.
    
    Args:
        text: Confirmation prompt
        default: Default value if user just presses Enter
        
    Returns:
        True if user confirms, False otherwise
    """
    return click.confirm(text, default=default)


def prompt(text: str, default: str = None, hide_input: bool = False) -> str:
    """
    Prompt user for input.
    
    Args:
        text: Prompt text
        default: Default value
        hide_input: Whether to hide input (for passwords)
        
    Returns:
        User input
    """
    return click.prompt(text, default=default, hide_input=hide_input)
