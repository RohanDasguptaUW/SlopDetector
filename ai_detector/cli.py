"""CLI entry point for SlopDetector."""

import glob
import json
import os
import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich import box

from . import ensemble, heatmap as heatmap_mod, report as report_mod
from .analyzers.base import AnalysisResult
from .analyzers.ela import ELAAnalyzer
from .analyzers.spectral import SpectralAnalyzer
from .analyzers.metadata import MetadataAnalyzer
from .analyzers.noise import NoiseAnalyzer

console = Console()

_SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


def _collect_images(paths: tuple[str, ...]) -> list[str]:
    images = []
    for p in paths:
        expanded = glob.glob(p, recursive=True)
        if not expanded:
            expanded = [p]
        for item in expanded:
            ip = Path(item)
            if ip.is_dir():
                for ext in _SUPPORTED_EXT:
                    images.extend(str(f) for f in ip.rglob(f"*{ext}"))
                    images.extend(str(f) for f in ip.rglob(f"*{ext.upper()}"))
            elif ip.suffix.lower() in _SUPPORTED_EXT:
                images.append(str(ip))
    # Deduplicate preserving order
    seen = set()
    unique = []
    for img in images:
        if img not in seen:
            seen.add(img)
            unique.append(img)
    return unique


def _analyse_image(
    image_path: str,
    use_claude: bool,
    model: str,
) -> tuple[list[AnalysisResult], dict]:
    results: list[AnalysisResult] = []

    analyzers = [ELAAnalyzer(), SpectralAnalyzer(), MetadataAnalyzer(), NoiseAnalyzer()]
    if use_claude:
        from .analyzers.claude import ClaudeAnalyzer
        analyzers.append(ClaudeAnalyzer(model=model))

    for analyzer in analyzers:
        try:
            r = analyzer.analyze(image_path)
        except Exception as exc:
            console.print(f"  [yellow]⚠ {analyzer.name} failed: {exc}[/yellow]")
            continue
        results.append(r)

    summary = ensemble.combine(results)
    return results, summary


def _print_results_table(image_path: str, results: list[AnalysisResult], summary: dict) -> None:
    table = Table(
        title=f"[bold cyan]{Path(image_path).name}[/bold cyan]",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("Analyzer", style="bold")
    table.add_column("AI %", justify="right")
    table.add_column("Confidence", justify="right")
    table.add_column("Key Indicators", overflow="fold", max_width=60)

    for r in results:
        pct_str = f"{r.ai_percentage:.1f}%"
        if r.ai_percentage >= 70:
            pct_str = f"[red]{pct_str}[/red]"
        elif r.ai_percentage >= 40:
            pct_str = f"[yellow]{pct_str}[/yellow]"
        else:
            pct_str = f"[green]{pct_str}[/green]"
        indicators = "; ".join(r.indicators[:2])
        table.add_row(r.analyzer, pct_str, f"{r.confidence:.2f}", indicators)

    # Ensemble row
    ens_pct = summary["ai_percentage"]
    ens_str = f"{ens_pct:.1f}%"
    if ens_pct >= 70:
        ens_str = f"[bold red]{ens_str}[/bold red]"
    elif ens_pct >= 40:
        ens_str = f"[bold yellow]{ens_str}[/bold yellow]"
    else:
        ens_str = f"[bold green]{ens_str}[/bold green]"

    table.add_row(
        "[bold]ENSEMBLE[/bold]",
        ens_str,
        f"[bold]{summary['confidence']:.2f}[/bold]",
        f"[bold]{summary['verdict']}[/bold]",
    )

    console.print(table)


@click.command("ai-detect")
@click.argument("images", nargs=-1, required=True, metavar="IMAGE [IMAGE ...]")
@click.option("--no-claude", is_flag=True, default=False, help="Skip Claude AI analyzer.")
@click.option("--heatmap", is_flag=True, default=False, help="Generate heatmap overlay PNG.")
@click.option("--report", is_flag=True, default=False, help="Generate self-contained HTML report.")
@click.option("--output", "-o", default=None, metavar="DIR", help="Output directory for generated files.")
@click.option("--json-out", is_flag=True, default=False, help="Print JSON summary to stdout.")
@click.option("--model", default="claude-sonnet-4-6", show_default=True, help="Claude model to use.")
def main(
    images: tuple[str, ...],
    no_claude: bool,
    heatmap: bool,
    report: bool,
    output: Optional[str],
    json_out: bool,
    model: str,
) -> None:
    """SlopDetector — estimate what percentage of an image is AI-generated.

    Pass one or more IMAGE paths, globs, or directories.

    Examples:

      ai-detect photo.jpg

      ai-detect *.png --heatmap --report -o results/

      ai-detect images/ --no-claude --json-out
    """
    use_claude = not no_claude
    image_list = _collect_images(images)

    if not image_list:
        console.print("[red]No supported images found.[/red]")
        sys.exit(1)

    out_dir = Path(output) if output else Path("slopdetector_output")
    out_dir.mkdir(parents=True, exist_ok=True)

    all_summaries = {}

    for img_path in image_list:
        console.rule(f"[bold blue]Analysing: {img_path}")
        try:
            results, summary = _analyse_image(img_path, use_claude, model)
        except Exception as exc:
            console.print(f"[red]Error processing {img_path}: {exc}[/red]")
            continue

        _print_results_table(img_path, results, summary)

        stem = Path(img_path).stem
        heatmap_path: Optional[str] = None

        # Generate heatmap
        if heatmap and summary.get("heatmap") is not None:
            hm_out = str(out_dir / f"{stem}_heatmap.png")
            try:
                heatmap_mod.generate(img_path, summary["heatmap"], summary["ai_percentage"], hm_out)
                console.print(f"  [cyan]Heatmap saved:[/cyan] {hm_out}")
                heatmap_path = hm_out
            except Exception as exc:
                console.print(f"  [yellow]Heatmap generation failed: {exc}[/yellow]")

        # Generate report
        if report:
            report_out = str(out_dir / f"{stem}_report.html")
            try:
                # serialise numpy arrays before JSON embedding
                import numpy as np
                safe_summary = {k: v for k, v in summary.items() if k != "heatmap"}
                report_mod.generate(safe_summary, img_path, heatmap_path, report_out)
                console.print(f"  [cyan]Report saved:[/cyan] {report_out}")
            except Exception as exc:
                console.print(f"  [yellow]Report generation failed: {exc}[/yellow]")

        # JSON output
        import numpy as np
        safe_summary = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                        for k, v in summary.items()}
        all_summaries[img_path] = safe_summary

        console.print()

    if json_out:
        console.print_json(json.dumps(all_summaries, indent=2))


if __name__ == "__main__":
    main()
