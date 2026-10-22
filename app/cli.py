"""Rich terminal interface: interactive menu plus command-line flags."""

from __future__ import annotations

import argparse
import importlib.util
import logging
import platform
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from app.agents.base import AgentContext
from app.config import MAX_WORDS, MIN_WORDS, ConfigError, Settings, load_settings
from app.demo.fixtures import DEMO_BANNER, FixtureSearchProvider
from app.graph import STAGES, run_workflow
from app.llm_client import OpenRouterClient, check_model_availability
from app.logging_config import setup_logging
from app.output_manager import OutputManager
from app.tools.web_search import ResearchService
from app.utils.text_utils import domain_of

logger = logging.getLogger(__name__)
console = Console()

DEFAULT_DEMO_TOPIC = "How Agentic AI Works"
STATUS_STYLE = {"supported": "green", "partially_supported": "yellow", "unsupported": "red",
                "contradictory": "magenta", "unverifiable": "bright_black"}

CONFIG_HELP = """[bold]Live mode needs a free OpenRouter model.[/bold]

1. Copy [cyan].env.example[/cyan] to [cyan].env[/cyan] in the project folder.
2. Create a key at [link]https://openrouter.ai/keys[/link] and set [cyan]OPENROUTER_API_KEY[/cyan].
3. Open [link]https://openrouter.ai/models[/link], filter by price "free", copy the ID of a
   currently available model and set [cyan]OPENROUTER_MODEL[/cyan] to it.
4. Optional: set [cyan]TAVILY_API_KEY[/cyan] (free tier at tavily.com); otherwise DuckDuckGo is used.

You can run [bold]python main.py --demo[/bold] to see the pipeline with simulated data."""


# --------------------------------------------------------------------------- helpers
def banner() -> None:
    console.print(Panel.fit("[bold cyan]BLOGFORGE AI[/bold cyan]\nMulti-Agent Blog Generation System",
                            border_style="cyan", padding=(1, 6)))


def demo_notice() -> None:
    console.print(Panel(f"[bold yellow]{DEMO_BANNER}[/bold yellow]", border_style="yellow"))


def clean_topic(topic: str | None) -> str | None:
    topic = " ".join((topic or "").split())
    return topic if 3 <= len(topic) <= 200 else None


def build_context(settings: Settings, demo: bool, topic: str) -> AgentContext | None:
    """Create the runtime context. Live mode never silently degrades to simulated data."""
    if demo:
        service = ResearchService([FixtureSearchProvider(topic)], max_results=settings.max_search_results,
                                  delay_seconds=0, simulated=True)
        return AgentContext(settings=settings, research_service=service, llm=None, demo=True)

    missing = settings.missing_for_live_mode()
    if missing:
        console.print(f"[red]Missing configuration: {', '.join(missing)}[/red]")
        console.print(Panel(CONFIG_HELP, title="How to configure live mode", border_style="red"))
        return None
    with console.status("Checking that the configured OpenRouter model is available and free..."):
        check = check_model_availability(settings.openrouter_model, timeout=settings.request_timeout)
    if check.reachable and not (check.available and check.is_free):
        console.print(f"[red]{check.message}[/red]")
        return None
    if not check.reachable:
        console.print(f"[yellow]Warning: {check.message} Continuing; the first LLM call will confirm access.[/yellow]")
    return AgentContext(settings=settings, research_service=ResearchService.from_settings(settings),
                        llm=OpenRouterClient.from_settings(settings), demo=False)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(Path.cwd()))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------------- generation
def generate(settings: Settings, topic: str, audience: str | None, tone: str | None, words: int,
             demo: bool, show: bool | None = None) -> bool:
    ctx = build_context(settings, demo, topic)
    if ctx is None:
        return False
    if demo:
        demo_notice()
    console.print(f"\n[bold]Topic:[/bold] {topic}   [bold]Mode:[/bold] {ctx.mode}   "
                  f"[bold]Model:[/bold] {ctx.model_name}\n")

    with console.status("Starting workflow...", spinner="dots") as status:
        def progress(stage: str, event: str, message: str) -> None:
            label = STAGES.get(stage, stage)
            if event == "start":
                status.update(f"[cyan]{label}[/cyan] working...")
            elif event == "done":
                console.print(f"  [green]✔[/green] [bold]{label}[/bold]: {message}")
            else:
                console.print(f"  [red]✖[/red] [bold]{label}[/bold] failed")

        ctx.progress = progress
        state = run_workflow(ctx, topic, audience, tone, words)
        status.update("Saving files...")
        paths = OutputManager(settings.blog_output_dir).save_run(state)
    console.print("  [green]✔[/green] [bold]Saving files[/bold]: done")

    for error in state.get("errors", []):
        console.print(f"[red]Error:[/red] {error}")
    show_result_summary(state, paths.files)
    final = state.get("final_blog")
    if final is None:
        console.print(f"[red]No blog was produced.[/red] Partial outputs: {paths.directory}")
        return False
    # if show is None:
    #     show = Confirm.ask("Display the generated blog now?", default=True)
    # if show:
    #     console.print(Markdown(final.markdown))
    console.print(Markdown(final.markdown))
    return True


def show_result_summary(state: dict, files: dict[str, Path]) -> None:
    final, package, report = state.get("final_blog"), state.get("research_package"), state.get("verification_report")
    table = Table(title="Run summary", show_header=False, title_style="bold")
    table.add_column(style="bold", no_wrap=True)
    table.add_column(overflow="fold")
    table.add_row("Status", state.get("execution_status", "unknown"))
    table.add_row("Mode", state.get("execution_metadata", {}).get("mode", "?"))
    if final:
        table.add_row("Blog title", final.title)
        table.add_row("Approx. word count", str(final.word_count))
    if package:
        table.add_row("Sources collected", f"{len(package.sources)} ({package.search_provider})")
    if report:
        problems = len(report.unsupported_claims) + len(report.unverifiable_claims) + len(report.contradictory_claims)
        table.add_row("Claims checked", str(report.claims_checked))
        table.add_row("Unsupported / unverifiable / contradictory", str(problems))
        table.add_row("Verification status", report.overall_status)
    table.add_row("Correction passes", str(state.get("retry_count", 0)))
    labels = {"blog": "Markdown file", "research": "Research JSON", "verification": "Verification JSON",
              "summary": "Summary JSON", "metadata": "Run metadata"}
    for key, label in labels.items():
        if key in files:
            table.add_row(label, _display_path(files[key]))
    console.print(table)
    warnings = state.get("warnings", [])
    if warnings:
        console.print(f"[yellow]{len(warnings)} warning(s):[/yellow]")
        for w in warnings[:8]:
            console.print(f"  [yellow]•[/yellow] {w}")
        if len(warnings) > 8:
            console.print("  … see run_metadata.json for the full list.")


# --------------------------------------------------------------------------- viewers
def view_previous(settings: Settings) -> None:
    runs = OutputManager(settings.blog_output_dir).list_runs()
    if not runs:
        console.print("[yellow]No saved runs yet.[/yellow]")
        return
    table = Table(title="Previously generated blogs")
    for col in ("#", "Folder", "Title", "Mode", "Status"):
        table.add_column(col)
    for i, run in enumerate(runs[:30], start=1):
        table.add_row(str(i), run.directory.name, run.title or "-", run.mode, run.status)
    console.print(table)
    choice = IntPrompt.ask("Enter a number to display the blog (0 to go back)", default=0)
    if 1 <= choice <= min(len(runs), 30):
        blog = runs[choice - 1].directory / "blog.md"
        if blog.exists():
            console.print(Markdown(blog.read_text(encoding="utf-8")))
        else:
            console.print("[yellow]That run did not produce a blog (see run_metadata.json).[/yellow]")


def view_latest_reports(settings: Settings) -> None:
    manager = OutputManager(settings.blog_output_dir)
    run = manager.latest_run()
    if run is None:
        console.print("[yellow]No saved runs yet.[/yellow]")
        return
    console.print(f"[bold]Latest run:[/bold] {run.directory}")
    research = manager.load_json(run.directory, "research")
    verification = manager.load_json(run.directory, "verification")
    if research:
        if research.get("mode") == "demonstration":
            demo_notice()
        table = Table(title=f"Sources ({research.get('search_provider')}) - research {research.get('research_status')}")
        for col in ("ID", "Title", "Domain", "Type", "Published"):
            table.add_column(col, overflow="fold")
        for s in research.get("sources", []):
            table.add_row(s["source_id"], s["title"][:70], domain_of(s["url"]), s.get("source_type", ""),
                          s.get("publication_date") or "unknown")
        console.print(table)
    if verification:
        cov = verification.get("source_coverage", {})
        console.print(f"[bold]Verification:[/bold] {verification.get('overall_status')} - "
                      f"{verification.get('claims_checked')} claims, support ratio {cov.get('claim_support_ratio')}, "
                      f"source utilisation {cov.get('source_utilization_ratio')}")
        table = Table(title="Checked claims")
        for col in ("ID", "Status", "Claim", "Sources"):
            table.add_column(col, overflow="fold")
        groups = ("verified_claims", "partially_supported_claims", "unsupported_claims",
                  "contradictory_claims", "unverifiable_claims")
        for claim in sorted((c for g in groups for c in verification.get(g, [])), key=lambda c: int(c["claim_id"][1:])):
            style = STATUS_STYLE.get(claim["status"], "white")
            table.add_row(claim["claim_id"], f"[{style}]{claim['status']}[/{style}]", claim["claim_text"][:110],
                          ", ".join(claim["supporting_source_ids"]) or "-")
        console.print(table)
        for limitation in verification.get("limitations", []):
            console.print(f"  [bright_black]• {limitation}[/bright_black]")
    if not research and not verification:
        console.print("[yellow]The latest run has no research or verification files.[/yellow]")


def config_check(settings: Settings, online: bool = True) -> bool:
    table = Table(title="System configuration check")
    for col in ("Check", "Result", "Details"):
        table.add_column(col, overflow="fold")
    ok = "[green]OK[/green]"

    def row(name: str, good: bool | None, details: str) -> None:
        result = ok if good else ("[yellow]WARN[/yellow]" if good is None else "[red]MISSING[/red]")
        table.add_row(name, result, details)

    row("Python", sys.version_info >= (3, 11), platform.python_version())
    row("OPENROUTER_API_KEY", settings.openrouter_api_key is not None,
        "configured (value hidden)" if settings.openrouter_api_key else "not set")
    row("OPENROUTER_MODEL", settings.openrouter_model is not None, settings.openrouter_model or "not set")
    if settings.openrouter_model and online:
        with console.status("Checking model on OpenRouter..."):
            check = check_model_availability(settings.openrouter_model, timeout=settings.request_timeout)
        row("Model available & free", (check.available and check.is_free) if check.reachable else None, check.message)
    row("TAVILY_API_KEY", True if settings.tavily_configured else None,
        "configured (value hidden)" if settings.tavily_configured else "not set - DuckDuckGo fallback will be used")
    row("DuckDuckGo fallback", importlib.util.find_spec("ddgs") is not None, "ddgs package")
    try:
        settings.blog_output_dir.mkdir(parents=True, exist_ok=True)
        probe = settings.blog_output_dir / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        row("Output directory", True, str(settings.blog_output_dir))
    except OSError as exc:
        row("Output directory", False, f"{settings.blog_output_dir}: {exc}")
    table.add_row("Limits", "", f"results={settings.max_search_results}, queries={settings.max_research_queries}, "
                  f"words={settings.blog_word_count}, retries={settings.verification_retries}, "
                  f"timeout={settings.request_timeout}s, log={settings.log_level}")
    console.print(table)
    if not settings.llm_configured:
        console.print(Panel(CONFIG_HELP, title="Live mode is not configured", border_style="yellow"))
        console.print("Demonstration mode is available: [bold]python main.py --demo[/bold]")
    return settings.llm_configured


# --------------------------------------------------------------------------- interactive
def interactive_generate(settings: Settings) -> None:
    topic = None
    while topic is None:
        topic = clean_topic(Prompt.ask("[bold]Blog topic[/bold]"))
        if topic is None:
            console.print("[red]Please enter a topic between 3 and 200 characters.[/red]")
    audience = Prompt.ask("Target audience (optional)", default="general readers").strip() or None
    tone = Prompt.ask("Writing tone (optional)", default="informative and approachable").strip() or None
    while True:
        words = IntPrompt.ask("Approximate word count", default=settings.blog_word_count)
        if MIN_WORDS <= words <= MAX_WORDS:
            break
        console.print(f"[red]Word count must be between {MIN_WORDS} and {MAX_WORDS}.[/red]")

    demo = False
    if not settings.llm_configured:
        console.print(Panel(CONFIG_HELP, title="Live mode is not configured", border_style="yellow"))
        if not Confirm.ask("Run an explicit DEMONSTRATION with simulated data instead?", default=True):
            return
        demo = True
    generate(settings, topic, audience, tone, words, demo=demo)


def menu(settings: Settings) -> None:
    actions = {
        "1": ("Generate a new blog", lambda: interactive_generate(settings)),
        "2": ("View previously generated blogs", lambda: view_previous(settings)),
        "3": ("View the latest research and verification report", lambda: view_latest_reports(settings)),
        "4": ("Run a system configuration check", lambda: config_check(settings)),
        "5": ("Exit", None),
    }
    while True:
        console.print()
        for key, (label, _) in actions.items():
            console.print(f"  [bold cyan]{key}.[/bold cyan] {label}")
        choice = Prompt.ask("\nChoose an option", choices=list(actions), default="1")
        if choice == "5":
            console.print("Goodbye!")
            return
        try:
            actions[choice][1]()
        except KeyboardInterrupt:
            console.print("\n[yellow]Cancelled.[/yellow]")
        except EOFError:  # input stream closed (Ctrl+Z / Ctrl+D or piped input ended)
            console.print("\n[yellow]No more input; exiting.[/yellow]")
            return


# --------------------------------------------------------------------------- entry point
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="main.py", description="BlogForge AI - terminal multi-agent blog generator")
    parser.add_argument("--topic", help="generate a blog for this topic without the interactive menu")
    parser.add_argument("--audience", help="target audience (optional)")
    parser.add_argument("--tone", help="writing tone (optional)")
    parser.add_argument("--words", type=int, help=f"approximate word count ({MIN_WORDS}-{MAX_WORDS})")
    parser.add_argument("--demo", action="store_true",
                        help="explicit demonstration mode with simulated data (no API keys or network)")
    parser.add_argument("--check", action="store_true", help="run the configuration check and exit")
    parser.add_argument("--show", action="store_true", help="print the generated blog in the terminal")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass
    args = parse_args(argv)
    banner()
    try:
        settings = load_settings()
    except ConfigError as exc:
        console.print(f"[red]{exc}[/red]\nFix the values in your .env file (see .env.example).")
        return 2
    log_file = setup_logging(settings.log_level, settings.secret_values())
    logger.info("BlogForge started (log file %s)", log_file)

    try:
        if args.check:
            return 0 if config_check(settings) else 1
        if args.topic or args.demo:
            topic = clean_topic(args.topic or DEFAULT_DEMO_TOPIC)
            if topic is None:
                console.print("[red]--topic must be between 3 and 200 characters.[/red]")
                return 2
            words = args.words or settings.blog_word_count
            if not MIN_WORDS <= words <= MAX_WORDS:
                console.print(f"[red]--words must be between {MIN_WORDS} and {MAX_WORDS}.[/red]")
                return 2
            ok = generate(settings, topic, args.audience, args.tone, words, demo=args.demo, show=args.show)
            return 0 if ok else 1
        menu(settings)
        return 0
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        return 130
    except EOFError:
        console.print("\n[yellow]No more input; exiting.[/yellow]")
        return 0
