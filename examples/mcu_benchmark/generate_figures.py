#!/usr/bin/env python3
"""Generate figures for MCU Comparative Benchmark.

Creates visualization figures from benchmark CSV results.

Usage:
    python generate_figures.py           # Generate all figures
    python generate_figures.py --check   # Check if dependencies are installed
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Check for plotly before importing
try:
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("Error: plotly not installed. Run: pip install plotly kaleido")
    sys.exit(1)

try:
    import kaleido  # noqa: F401 - just checking import
except ImportError:
    print("Warning: kaleido not installed. PNG export may fail.")
    print("Run: pip install kaleido")

# Paths
SCRIPT_DIR = Path(__file__).parent
RESULTS_DIR = SCRIPT_DIR / "results"
FIGURES_DIR = SCRIPT_DIR / "figures"

# Run directory names — update CALIB_RUN_NAME after re-running probability_calibration
LATENCY_STD_RUN = "2026-02-16_170735_sweep"
CALIB_RUN_NAME = "2026-02-17_103624_calib_lu"
FULL_SWEEP_RUN = "2026-02-17_163643_sweep"

# Color schemes - consistent across all figures
PLATFORM_COLORS = {
    'host': '#94a3b8',            # Gray
    'renode_nrf52840': '#3b82f6', # Blue (Renode emulator)
    'native_sim': '#8b5cf6',      # Purple (native)
    'nrf52dk_nrf52832': '#10b981', # Green (hardware)
}

# Paul Tol colorblind-friendly palette (bright scheme)
MODEL_COLORS = {
    'gbt': '#4477AA',     # Blue
    'rf': '#EE6677',      # Red/Pink
}

PLATFORM_NAMES = {
    'host': 'Host (CFFI)',
    'renode_nrf52840': 'Renode nRF52840 (FPU)',
    'native_sim': 'Native Sim (FPU)',
    'nrf52dk_nrf52832': 'nRF52 DK (FPU)',
}

MODEL_NAMES = {
    'gbt': 'GBT',
    'rf': 'RF',
}

SIZE_NAMES = {
    'small': 'Small (5 trees)',
    'medium': 'Medium (10 trees)',
    'large': 'Large (20 trees)',
}


def load_csv(filename: str = "mcu_inference_timing.csv") -> pd.DataFrame:
    """Load benchmark results CSV."""
    filepath = RESULTS_DIR / filename
    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return pd.DataFrame()
    return pd.read_csv(filepath)


def save_figure(fig, filename: str, width: int = 800, height: int = 600, scale: int = 2):
    """Save figure as PNG."""
    FIGURES_DIR.mkdir(exist_ok=True)
    filepath = FIGURES_DIR / filename
    fig.write_image(str(filepath), width=width, height=height, scale=scale)
    print(f"Saved: {filepath}")


def _runs_dir(results_dir: Path) -> Path:
    """Return the runs/ directory given a results_dir.

    Works for both:
    - results_dir = runs/<timestamp>/results  -> returns runs/
    - results_dir = mcu_benchmark/results (default mode) -> returns mcu_benchmark/runs/
    """
    candidate = results_dir.parent.parent
    if candidate.name == "runs":
        return candidate
    return SCRIPT_DIR / "runs"


def get_model_size(model_name: str) -> str:
    """Extract size category from model name."""
    if 'small' in model_name:
        return 'small'
    elif 'medium' in model_name:
        return 'medium'
    elif 'large' in model_name:
        return 'large'
    return 'unknown'


def figure_platform_comparison():
    """Figure 1: Grouped bar comparing Host vs Renode vs Hardware timing."""
    df = load_csv()
    if df.empty:
        print("Skipping platform_comparison figure: no data")
        return

    # Get unique platforms and model sizes
    df['size'] = df['model_name'].apply(get_model_size)
    platforms = df['platform'].unique()
    sizes = ['small', 'medium', 'large']

    fig = go.Figure()

    # Group by model type and platform
    x_labels = []
    for size in sizes:
        for model_type in ['gbt', 'rf']:
            x_labels.append(f"{MODEL_NAMES[model_type]}\n{SIZE_NAMES.get(size, size)}")

    bar_width = 0.8 / len(platforms)
    x_positions = list(range(len(x_labels)))

    for idx, platform in enumerate(platforms):
        values = []
        for size in sizes:
            for model_type in ['gbt', 'rf']:
                data = df[(df['platform'] == platform) &
                          (df['model_type'] == model_type) &
                          (df['size'] == size)]
                if not data.empty:
                    values.append(data['avg_ns'].mean())
                else:
                    values.append(0)

        offset = (idx - (len(platforms) - 1) / 2) * bar_width

        fig.add_trace(go.Bar(
            x=[x + offset for x in x_positions],
            y=values,
            name=PLATFORM_NAMES.get(platform, platform),
            marker_color=PLATFORM_COLORS.get(platform, '#888888'),
            width=bar_width,
            text=[f"{v:.0f}" if v > 0 else "" for v in values],
            textposition="outside",
        ))

    fig.update_layout(
        title=dict(
            text="MCU Platform Comparison: Inference Timing",
            x=0.5,
            font=dict(size=16),
        ),
        xaxis=dict(
            tickmode="array",
            tickvals=x_positions,
            ticktext=x_labels,
            title="Model Configuration",
        ),
        yaxis=dict(
            title="Inference Time (μs)",
            type="log",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        barmode="group",
        font=dict(size=12),
        margin=dict(t=80, b=100, l=60, r=40),
    )

    save_figure(fig, "mcu_platform_comparison.png", width=1000, height=600, scale=1)


def figure_cycles_vs_accuracy():
    """Figure 2: Scatter plot showing Pareto frontier of cycles vs accuracy."""
    df = load_csv()
    if df.empty:
        print("Skipping cycles_vs_accuracy figure: no data")
        return

    # Filter to platforms with cycle counts (not host)
    df = df[df['platform'] != 'host']
    if df.empty:
        print("Skipping cycles_vs_accuracy figure: no MCU data")
        return

    fig = go.Figure()

    for platform in df['platform'].unique():
        platform_df = df[df['platform'] == platform]

        for model_type in ['gbt', 'rf']:
            type_df = platform_df[platform_df['model_type'] == model_type]

            if type_df.empty:
                continue

            # Create marker symbols based on platform
            symbol = 'circle' if 'renode' in platform else 'diamond'

            fig.add_trace(go.Scatter(
                x=type_df['avg_cycles'],
                y=type_df['accuracy'] * 100,
                mode='markers',
                name=f"{MODEL_NAMES[model_type]} ({PLATFORM_NAMES.get(platform, platform)})",
                marker=dict(
                    color=MODEL_COLORS[model_type],
                    size=12,
                    symbol=symbol,
                    line=dict(width=1, color='white'),
                ),
                text=type_df['model_name'],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "Cycles: %{x:,.0f}<br>"
                    "Accuracy: %{y:.1f}%<br>"
                    "<extra></extra>"
                ),
            ))

    fig.update_layout(
        title=dict(
            text="MCU Cycles vs Accuracy Trade-off",
            x=0.5,
            font=dict(size=16),
        ),
        xaxis=dict(
            title="Average CPU Cycles",
            type="log",
        ),
        yaxis=dict(
            title="Accuracy (%)",
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        font=dict(size=12),
        margin=dict(t=80, b=60, l=60, r=40),
    )

    save_figure(fig, "mcu_cycles_vs_accuracy.png", width=800, height=600, scale=1)


def figure_gbt_vs_rf():
    """Figure 3: 2x2 grid comparing GBT vs RF per model size and platform."""
    df = load_csv()
    if df.empty:
        print("Skipping gbt_vs_rf figure: no data")
        return

    df['size'] = df['model_name'].apply(get_model_size)

    # Filter to MCU platforms only
    df_mcu = df[df['platform'] != 'host']
    if df_mcu.empty:
        print("Skipping gbt_vs_rf figure: no MCU data")
        return

    platforms = df_mcu['platform'].unique()
    sizes = ['small', 'medium', 'large']

    # Create 2x2 subplot (platforms x metrics)
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Inference Time (μs)",
            "CPU Cycles",
            "Timing by Model Size",
            "Cycles by Model Size",
        ],
        horizontal_spacing=0.12,
        vertical_spacing=0.15,
    )

    # Top row: grouped bars by platform for inference time and cycles
    for col, metric in enumerate(['avg_ns', 'avg_cycles'], 1):
        for model_type in ['gbt', 'rf']:
            values = []
            platform_labels = []

            for platform in platforms:
                data = df_mcu[(df_mcu['platform'] == platform) &
                              (df_mcu['model_type'] == model_type)]
                if not data.empty:
                    values.append(data[metric].mean())
                    platform_labels.append(PLATFORM_NAMES.get(platform, platform))
                else:
                    values.append(0)
                    platform_labels.append(PLATFORM_NAMES.get(platform, platform))

            fig.add_trace(
                go.Bar(
                    x=platform_labels,
                    y=values,
                    name=MODEL_NAMES[model_type],
                    marker_color=MODEL_COLORS[model_type],
                    showlegend=(col == 1),
                    legendgroup=model_type,
                ),
                row=1, col=col
            )

    # Bottom row: by model size
    for col, metric in enumerate(['avg_ns', 'avg_cycles'], 1):
        for model_type in ['gbt', 'rf']:
            values = []

            for size in sizes:
                data = df_mcu[(df_mcu['size'] == size) &
                              (df_mcu['model_type'] == model_type)]
                if not data.empty:
                    values.append(data[metric].mean())
                else:
                    values.append(0)

            fig.add_trace(
                go.Bar(
                    x=[SIZE_NAMES.get(s, s) for s in sizes],
                    y=values,
                    name=MODEL_NAMES[model_type],
                    marker_color=MODEL_COLORS[model_type],
                    showlegend=False,
                    legendgroup=model_type,
                ),
                row=2, col=col
            )

    fig.update_layout(
        title=dict(
            text="GBT vs RF: MCU Performance Comparison",
            x=0.5,
            font=dict(size=16),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        barmode="group",
        font=dict(size=12),
        margin=dict(t=100, b=60, l=60, r=40),
    )

    save_figure(fig, "mcu_gbt_vs_rf.png", width=900, height=700, scale=1)


def figure_method_comparison():
    """Figure 4: Grouped bar comparing Inline vs Loadable methods."""
    df = load_csv()
    if df.empty:
        print("Skipping method_comparison figure: no data")
        return

    # Check if method column has multiple values
    if 'method' not in df.columns or df['method'].nunique() <= 1:
        print("Skipping method_comparison figure: only one method in data")
        return

    # Filter to hardware only for method comparison
    df_hw = df[df['platform_type'] == 'hardware']
    if df_hw.empty:
        print("Skipping method_comparison figure: no hardware data")
        return

    df_hw['size'] = df_hw['model_name'].apply(get_model_size)
    methods = df_hw['method'].unique()
    sizes = ['small', 'medium', 'large']

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["GBT", "RF"],
        horizontal_spacing=0.15,
    )

    bar_width = 0.35

    for col, model_type in enumerate(['gbt', 'rf'], 1):
        for idx, method in enumerate(methods):
            values = []
            for size in sizes:
                data = df_hw[(df_hw['size'] == size) &
                             (df_hw['model_type'] == model_type) &
                             (df_hw['method'] == method)]
                if not data.empty:
                    values.append(data['avg_ns'].mean())
                else:
                    values.append(0)

            offset = -bar_width/2 if idx == 0 else bar_width/2

            fig.add_trace(
                go.Bar(
                    x=[i + offset for i in range(len(sizes))],
                    y=values,
                    name=method.capitalize(),
                    marker_color='#3b82f6' if method == 'inline' else '#f59e0b',
                    width=bar_width,
                    showlegend=(col == 1),
                    legendgroup=method,
                ),
                row=1, col=col
            )

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(sizes))),
        ticktext=[SIZE_NAMES.get(s, s) for s in sizes],
    )
    fig.update_yaxes(title_text="Inference Time (μs)", col=1)

    fig.update_layout(
        title=dict(
            text="Method Comparison: Inline vs Loadable on Hardware",
            x=0.5,
            font=dict(size=16),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="center",
            x=0.5,
        ),
        barmode="group",
        font=dict(size=12),
        margin=dict(t=100, b=60, l=60, r=40),
    )

    save_figure(fig, "mcu_method_comparison.png", width=800, height=450, scale=1)


def figure_timing_breakdown():
    """Figure 5: Timing breakdown by model size across all platforms."""
    df = load_csv()
    if df.empty:
        print("Skipping timing_breakdown figure: no data")
        return

    df['size'] = df['model_name'].apply(get_model_size)
    sizes = ['small', 'medium', 'large']

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[SIZE_NAMES.get(s, s) for s in sizes],
        horizontal_spacing=0.1,
    )

    platforms = df['platform'].unique()

    for col, size in enumerate(sizes, 1):
        size_df = df[df['size'] == size]

        for model_type in ['gbt', 'rf']:
            values = []
            for platform in platforms:
                data = size_df[(size_df['platform'] == platform) &
                               (size_df['model_type'] == model_type)]
                if not data.empty:
                    values.append(data['avg_ns'].mean())
                else:
                    values.append(0)

            fig.add_trace(
                go.Bar(
                    x=[PLATFORM_NAMES.get(p, p) for p in platforms],
                    y=values,
                    name=MODEL_NAMES[model_type],
                    marker_color=MODEL_COLORS[model_type],
                    showlegend=(col == 1),
                    legendgroup=model_type,
                ),
                row=1, col=col
            )

    fig.update_yaxes(title_text="Inference Time (μs)", type="log", col=1)
    fig.update_yaxes(type="log", col=2)
    fig.update_yaxes(type="log", col=3)

    fig.update_layout(
        title=dict(
            text="Timing Breakdown by Model Size",
            x=0.5,
            font=dict(size=16),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="center",
            x=0.5,
        ),
        barmode="group",
        font=dict(size=12),
        margin=dict(t=100, b=60, l=60, r=40),
    )

    save_figure(fig, "mcu_timing_breakdown.png", width=1000, height=450, scale=1)


def figure_fpu_comparison():
    """Figure 6: FPU Impact - GBT vs RF on FPU vs no-FPU platforms.

    Shows that GBT benefits dramatically from hardware FPU (expf acceleration)
    while RF performance is unaffected (integer-only voting).
    """
    df = load_csv()
    if df.empty:
        print("Skipping fpu_comparison figure: no data")
        return

    # Filter to emulator platforms only (renode vs native_sim)
    fpu_platforms = ['renode_nrf52840', 'native_sim']
    df = df[df['platform'].isin(fpu_platforms)]
    if len(df['platform'].unique()) < 2:
        print("Skipping fpu_comparison figure: need both FPU and no-FPU platforms")
        return

    df['size'] = df['model_name'].apply(get_model_size)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["GBT (uses expf → FPU matters)", "RF (integer voting → FPU irrelevant)"],
        horizontal_spacing=0.15,
    )

    sizes = ['small', 'medium', 'large']
    bar_width = 0.35

    for col, model_type in enumerate(['gbt', 'rf'], 1):
        for idx, platform in enumerate(fpu_platforms):
            values = []
            for size in sizes:
                data = df[(df['size'] == size) &
                          (df['model_type'] == model_type) &
                          (df['platform'] == platform)]
                if not data.empty:
                    values.append(data['avg_ns'].mean())
                else:
                    values.append(0)

            offset = -bar_width/2 if idx == 0 else bar_width/2

            fig.add_trace(
                go.Bar(
                    x=[i + offset for i in range(len(sizes))],
                    y=values,
                    name=PLATFORM_NAMES.get(platform, platform),
                    marker_color=PLATFORM_COLORS.get(platform, '#888888'),
                    width=bar_width,
                    showlegend=(col == 1),
                    legendgroup=platform,
                    text=[f"{v:.0f}" if v > 0 else "" for v in values],
                    textposition="outside",
                ),
                row=1, col=col
            )

    fig.update_xaxes(
        tickmode="array",
        tickvals=list(range(len(sizes))),
        ticktext=[SIZE_NAMES.get(s, s) for s in sizes],
    )
    fig.update_yaxes(title_text="Inference Time (μs)", type="log", col=1)
    fig.update_yaxes(type="log", col=2)

    fig.update_layout(
        title=dict(
            text="FPU Impact: GBT Benefits from Hardware FPU, RF Unaffected",
            x=0.5,
            font=dict(size=16),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.08,
            xanchor="center",
            x=0.5,
        ),
        barmode="group",
        font=dict(size=12),
        margin=dict(t=100, b=60, l=60, r=40),
        annotations=[
            dict(
                text="GBT: ~5-10x faster with FPU (hardware expf)",
                xref="paper", yref="paper",
                x=0.25, y=-0.15,
                showarrow=False,
                font=dict(size=10, color="#666"),
            ),
            dict(
                text="RF: Similar speed (no floating-point ops)",
                xref="paper", yref="paper",
                x=0.75, y=-0.15,
                showarrow=False,
                font=dict(size=10, color="#666"),
            ),
        ],
    )

    save_figure(fig, "mcu_fpu_comparison.png", width=900, height=500, scale=1)


def figure_sample_efficiency_classification(results_dir: Path, figures_dir: Path):
    """Generate sample efficiency figure for classification datasets."""
    csv_path = results_dir / "sample_efficiency.csv"
    if not csv_path.exists():
        print("Skipping sample_efficiency_classification figure: no data")
        return

    df = pd.read_csv(csv_path)
    df = df[df['task'] == 'classification']
    if df.empty:
        return

    datasets = ['embedded_synth', 'sonar', 'breast_cancer', 'wine', 'iris', 'digits']
    dataset_labels = {
        'embedded_synth': 'Embedded Synth',
        'sonar': 'Sonar',
        'breast_cancer': 'Breast Cancer',
        'wine': 'Wine',
        'iris': 'Iris',
        'digits': 'Digits',
    }

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[dataset_labels.get(d, d) for d in datasets],
        horizontal_spacing=0.08,
        vertical_spacing=0.18,
    )

    shown_in_legend = set()
    for idx, dataset in enumerate(datasets):
        row = idx // 3 + 1
        col = idx % 3 + 1
        ds_df = df[df['dataset'] == dataset]

        for model_type, line_color, dash in [
            ('gbt', MODEL_COLORS['gbt'], 'solid'),
            ('rf', MODEL_COLORS['rf'], 'dash'),
        ]:
            mt_df = ds_df[ds_df['model_type'] == model_type].sort_values('n_estimators')
            if mt_df.empty:
                continue
            show = model_type not in shown_in_legend
            if show:
                shown_in_legend.add(model_type)
            fig.add_trace(go.Scatter(
                x=mt_df['n_estimators'],
                y=mt_df['score'],
                mode='lines+markers',
                name=MODEL_NAMES.get(model_type, model_type),
                showlegend=show,
                line=dict(color=line_color, dash=dash),
                marker=dict(color=line_color, size=7),
            ), row=row, col=col)

    fig.update_xaxes(title_text="n_estimators")
    fig.update_yaxes(title_text="Accuracy", col=1)
    fig.update_layout(
        title=dict(text="Sample Efficiency: Classification", x=0.5, font=dict(size=16)),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        margin=dict(t=100, b=60, l=60, r=40),
        font=dict(size=11),
    )

    figures_dir.mkdir(exist_ok=True)
    filepath = figures_dir / "sample_efficiency_classification.png"
    fig.write_image(str(filepath), width=1200, height=700, scale=2)
    print(f"Saved: {filepath}")


def figure_sample_efficiency_regression(results_dir: Path, figures_dir: Path):
    """Generate sample efficiency figure for regression datasets."""
    csv_path = results_dir / "sample_efficiency.csv"
    if not csv_path.exists():
        print("Skipping sample_efficiency_regression figure: no data")
        return

    df = pd.read_csv(csv_path)
    df = df[df['task'] == 'regression']
    if df.empty:
        return

    datasets = ['additive_synth', 'california', 'diabetes']
    dataset_labels = {
        'additive_synth': 'Additive Synth',
        'california': 'California Housing',
        'diabetes': 'Diabetes',
    }

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[dataset_labels.get(d, d) for d in datasets],
        horizontal_spacing=0.10,
    )

    shown_in_legend = set()
    for idx, dataset in enumerate(datasets):
        col = idx + 1
        ds_df = df[df['dataset'] == dataset]

        for model_type, line_color, dash in [
            ('gbt', MODEL_COLORS['gbt'], 'solid'),
            ('rf', MODEL_COLORS['rf'], 'dash'),
        ]:
            mt_df = ds_df[ds_df['model_type'] == model_type].sort_values('n_estimators')
            if mt_df.empty:
                continue
            show = model_type not in shown_in_legend
            if show:
                shown_in_legend.add(model_type)
            fig.add_trace(go.Scatter(
                x=mt_df['n_estimators'],
                y=mt_df['score'],
                mode='lines+markers',
                name=MODEL_NAMES.get(model_type, model_type),
                showlegend=show,
                line=dict(color=line_color, dash=dash),
                marker=dict(color=line_color, size=7),
            ), row=1, col=col)

    fig.update_xaxes(title_text="n_estimators")
    fig.update_yaxes(title_text="neg MSE (higher=better)", col=1)
    fig.update_layout(
        title=dict(text="Sample Efficiency: Regression", x=0.5, font=dict(size=16)),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
        margin=dict(t=100, b=60, l=60, r=40),
        font=dict(size=11),
    )

    figures_dir.mkdir(exist_ok=True)
    filepath = figures_dir / "sample_efficiency_regression.png"
    fig.write_image(str(filepath), width=1000, height=400, scale=2)
    print(f"Saved: {filepath}")


def figure_pareto_efficiency(results_dir: Path, figures_dir: Path):
    """Generate Pareto efficiency figure: accuracy vs flash, one subplot per dataset."""
    csv_path = results_dir / "pareto_efficiency.csv"
    if not csv_path.exists():
        print("Skipping pareto_efficiency figure: no data")
        return

    df = pd.read_csv(csv_path)
    df = df[(df['task'] == 'classification') & (df['platform'] == 'host')]
    if df.empty:
        return

    datasets = ['embedded_synth', 'sonar', 'breast_cancer', 'wine', 'iris', 'digits']
    dataset_labels = {
        'embedded_synth': 'Embedded Synth',
        'sonar': 'Sonar',
        'breast_cancer': 'Breast Cancer',
        'wine': 'Wine',
        'iris': 'Iris',
        'digits': 'Digits',
    }

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[dataset_labels.get(d, d) for d in datasets],
        horizontal_spacing=0.08,
        vertical_spacing=0.18,
    )

    shown_in_legend = set()
    for idx, dataset in enumerate(datasets):
        row = idx // 3 + 1
        col = idx % 3 + 1
        ds_df = df[df['dataset'] == dataset]

        for model_type, color in [('gbt', MODEL_COLORS['gbt']), ('rf', MODEL_COLORS['rf'])]:
            mt_df = ds_df[ds_df['model_type'] == model_type]
            if mt_df.empty:
                continue

            # Faded scatter: all configs
            show_scatter = f"{model_type}_scatter" not in shown_in_legend
            if show_scatter:
                shown_in_legend.add(f"{model_type}_scatter")
            fig.add_trace(go.Scatter(
                x=mt_df['flash_bytes'] / 1024,
                y=mt_df['score'],
                mode='markers',
                name=MODEL_NAMES.get(model_type, model_type),
                showlegend=show_scatter,
                marker=dict(color=color, size=6, opacity=0.3,
                            symbol='circle' if model_type == 'gbt' else 'diamond'),
                legendgroup=model_type,
            ), row=row, col=col)

            # Bold Pareto frontier
            if 'is_pareto' in mt_df.columns:
                pareto_df = mt_df[mt_df['is_pareto']].sort_values('flash_bytes')
                if len(pareto_df) > 0:
                    show_pareto = f"{model_type}_pareto" not in shown_in_legend
                    if show_pareto:
                        shown_in_legend.add(f"{model_type}_pareto")
                    fig.add_trace(go.Scatter(
                        x=pareto_df['flash_bytes'] / 1024,
                        y=pareto_df['score'],
                        mode='lines+markers',
                        name=f"{MODEL_NAMES.get(model_type, model_type)} Pareto",
                        showlegend=show_pareto,
                        line=dict(color=color, width=2),
                        marker=dict(color=color, size=9,
                                    symbol='circle' if model_type == 'gbt' else 'diamond'),
                        legendgroup=f"{model_type}_pareto",
                    ), row=row, col=col)

    fig.update_xaxes(title_text="Flash (KB)")
    fig.update_yaxes(title_text="Accuracy", col=1)
    fig.update_layout(
        title=dict(text="Pareto Efficiency: Accuracy vs Flash Size", x=0.5, font=dict(size=16)),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        margin=dict(t=100, b=60, l=60, r=40),
        font=dict(size=11),
    )

    figures_dir.mkdir(exist_ok=True)
    filepath = figures_dir / "pareto_efficiency.png"
    fig.write_image(str(filepath), width=1200, height=700, scale=2)
    print(f"Saved: {filepath}")


def figure_size_constrained(results_dir: Path, figures_dir: Path):
    """Generate size-constrained figures: best accuracy/R² at fixed flash budgets."""
    csv_path = results_dir / "size_constrained.csv"
    if not csv_path.exists():
        print("Skipping size_constrained figure: no data")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    budget_labels = {2048: "2KB", 4096: "4KB", 8192: "8KB", 16384: "16KB", 32768: "32KB", 65536: "64KB"}
    budgets = [2048, 4096, 8192, 16384, 32768, 65536]
    budget_names = [budget_labels[b] for b in budgets]

    def _make_size_constrained_fig(task_df, datasets, dataset_labels, y_title, rows, cols):
        fig = make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[dataset_labels.get(d, d) for d in datasets],
            horizontal_spacing=0.08,
            vertical_spacing=0.20,
        )
        shown_in_legend = set()
        for idx, dataset in enumerate(datasets):
            row = idx // cols + 1
            col = idx % cols + 1
            ds_df = task_df[task_df['dataset'] == dataset]

            for model_type, color in [('gbt', MODEL_COLORS['gbt']), ('rf', MODEL_COLORS['rf'])]:
                mt_df = ds_df[ds_df['model_type'] == model_type]
                scores = []
                texts = []
                for b in budgets:
                    row_b = mt_df[mt_df['budget'] == b]
                    if row_b.empty:
                        scores.append(0.0)
                        texts.append("")
                    else:
                        best = row_b['score'].max()
                        scores.append(best)
                        texts.append(f"{best:.2f}")

                show = model_type not in shown_in_legend
                if show:
                    shown_in_legend.add(model_type)
                fig.add_trace(go.Bar(
                    x=budget_names,
                    y=scores,
                    name=MODEL_NAMES.get(model_type, model_type),
                    showlegend=show,
                    marker_color=color,
                    text=texts,
                    textposition='outside',
                    legendgroup=model_type,
                    textfont=dict(size=9),
                ), row=row, col=col)

        fig.update_xaxes(title_text="Flash Budget")
        fig.update_yaxes(title_text=y_title, col=1)
        fig.update_layout(
            barmode='group',
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
            margin=dict(t=100, b=60, l=60, r=40),
            font=dict(size=11),
        )
        return fig

    # Classification
    cls_df = df[df['task'] == 'classification']
    if not cls_df.empty:
        cls_datasets = ['embedded_synth', 'sonar', 'breast_cancer', 'wine', 'iris', 'digits']
        cls_datasets = [d for d in cls_datasets if d in cls_df['dataset'].unique()]
        cls_labels = {
            'embedded_synth': 'Embedded Synth', 'sonar': 'Sonar',
            'breast_cancer': 'Breast Cancer', 'wine': 'Wine',
            'iris': 'Iris', 'digits': 'Digits',
        }
        fig_cls = _make_size_constrained_fig(cls_df, cls_datasets, cls_labels, "Accuracy", 2, 3)
        fig_cls.update_layout(
            title=dict(text="Size-Constrained: Best Accuracy at Fixed Flash Budget",
                       x=0.5, font=dict(size=16)),
        )
        figures_dir.mkdir(exist_ok=True)
        filepath = figures_dir / "size_constrained_classification.png"
        fig_cls.write_image(str(filepath), width=1200, height=700, scale=2)
        print(f"Saved: {filepath}")

    # Regression
    reg_df = df[df['task'] == 'regression']
    if not reg_df.empty:
        reg_datasets = ['additive_synth', 'california', 'diabetes']
        reg_datasets = [d for d in reg_datasets if d in reg_df['dataset'].unique()]
        reg_labels = {
            'additive_synth': 'Additive Synth',
            'california': 'California Housing',
            'diabetes': 'Diabetes',
        }
        fig_reg = _make_size_constrained_fig(reg_df, reg_datasets, reg_labels, "R²", 1, 3)
        fig_reg.update_layout(
            title=dict(text="Size-Constrained: Best R² at Fixed Flash Budget",
                       x=0.5, font=dict(size=16)),
        )
        figures_dir.mkdir(exist_ok=True)
        filepath = figures_dir / "size_constrained_regression.png"
        fig_reg.write_image(str(filepath), width=1000, height=400, scale=2)
        print(f"Saved: {filepath}")


def figure_regression_accuracy(results_dir: Path, figures_dir: Path):
    """Generate regression accuracy figure: R² vs n_estimators, GBT vs RF."""
    csv_path = results_dir / "regression_accuracy.csv"
    if not csv_path.exists():
        print("Skipping regression_accuracy figure: no data")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    datasets = ['additive_synth', 'california', 'diabetes']
    dataset_labels = {
        'additive_synth': 'Additive Synth',
        'california': 'California Housing',
        'diabetes': 'Diabetes',
    }

    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[dataset_labels.get(d, d) for d in datasets],
        horizontal_spacing=0.10,
    )

    depth_dashes = {3: 'solid', 4: 'dot', 5: 'dash'}
    shown_in_legend = set()

    for idx, dataset in enumerate(datasets):
        col = idx + 1
        ds_df = df[df['dataset'] == dataset]

        for model_type, color in [('gbt', MODEL_COLORS['gbt']), ('rf', MODEL_COLORS['rf'])]:
            mt_df = ds_df[ds_df['model_type'] == model_type]
            if mt_df.empty:
                continue

            for depth in sorted(mt_df['max_depth'].unique()):
                d_df = mt_df[mt_df['max_depth'] == depth]
                # Select best lr per n_estimators (max r2_emlearn)
                best = d_df.loc[d_df.groupby('n_estimators')['r2_emlearn'].idxmax()]
                best = best.sort_values('n_estimators')

                legend_key = f"{model_type}_d{depth}"
                show = legend_key not in shown_in_legend
                if show:
                    shown_in_legend.add(legend_key)
                fig.add_trace(go.Scatter(
                    x=best['n_estimators'],
                    y=best['r2_emlearn'],
                    mode='lines+markers',
                    name=f"{MODEL_NAMES.get(model_type, model_type)} d={depth}",
                    showlegend=show,
                    line=dict(color=color, dash=depth_dashes.get(depth, 'solid')),
                    marker=dict(color=color, size=7),
                    legendgroup=legend_key,
                ), row=1, col=col)

    fig.update_xaxes(title_text="n_estimators")
    fig.update_yaxes(title_text="R² (emlearn)", col=1)
    fig.update_layout(
        title=dict(text="Regression Accuracy: R² vs Model Size", x=0.5, font=dict(size=16)),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
        margin=dict(t=100, b=60, l=60, r=40),
        font=dict(size=11),
    )

    figures_dir.mkdir(exist_ok=True)
    filepath = figures_dir / "regression_accuracy.png"
    fig.write_image(str(filepath), width=1000, height=400, scale=2)
    print(f"Saved: {filepath}")


def figure_calibration_comparison(results_dir: Path, figures_dir: Path):
    """Figure: GBT vs RF calibration quality."""
    runs_parent = _runs_dir(results_dir)
    calib_run = runs_parent / CALIB_RUN_NAME
    csv_path = calib_run / "results" / "probability_calibration.csv"
    if not csv_path.exists():
        # Fallback to results_dir
        csv_path = results_dir / 'probability_calibration.csv'
    if not csv_path.exists():
        print("Skipping calibration_comparison figure: no data")
        return

    df = pd.read_csv(csv_path)
    if df.empty:
        return

    # Get best Brier score per dataset/model (lowest is best)
    best = df.groupby(['dataset', 'model_type'])['brier_score'].min().unstack()

    # Order datasets by complexity (simple to complex)
    datasets = ['iris', 'wine', 'breast_cancer', 'sonar', 'embedded_synth', 'digits']
    # Filter to only datasets that exist
    datasets = [ds for ds in datasets if ds in best.index]

    if not datasets:
        print("Skipping calibration_comparison figure: no matching datasets")
        return

    gbt_scores = [best.loc[ds, 'gbt'] if 'gbt' in best.columns else 0 for ds in datasets]
    rf_scores = [best.loc[ds, 'rf'] if 'rf' in best.columns else 0 for ds in datasets]

    # Capitalize dataset names for display
    dataset_labels = [ds.replace('_', ' ').title() for ds in datasets]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name='RF',
        x=dataset_labels,
        y=rf_scores,
        marker_color=MODEL_COLORS['rf'],
        text=[f'{s:.3f}' for s in rf_scores],
        textposition='outside',
    ))

    fig.add_trace(go.Bar(
        name='GBT',
        x=dataset_labels,
        y=gbt_scores,
        marker_color=MODEL_COLORS['gbt'],
        text=[f'{s:.3f}' for s in gbt_scores],
        textposition='outside',
    ))

    fig.update_layout(
        title=dict(
            text='Probability Calibration Quality (Lower is Better)',
            x=0.5,
            font=dict(size=16),
        ),
        xaxis_title='Dataset',
        yaxis_title='Brier Score',
        barmode='group',
        yaxis=dict(range=[0, max(gbt_scores + rf_scores) * 1.2]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        font=dict(size=12),
        margin=dict(t=80, b=60, l=60, r=40),
    )

    figures_dir.mkdir(exist_ok=True)
    filepath = figures_dir / 'calibration_comparison.png'
    fig.write_image(str(filepath), width=800, height=600, scale=2)
    print(f"Saved: {filepath}")


def figure_binary_classification(results_dir: Path, figures_dir: Path):
    """Figure: Binary classification with RF vs GBT speed comparison."""
    runs_parent = _runs_dir(results_dir)

    std_run = runs_parent / LATENCY_STD_RUN

    datasets = {
        'breast_cancer': 'Breast Cancer (30 features, 2 classes)',
        'sonar': 'Sonar (60 features, 2 classes)',
    }
    configs = [(3, 3), (3, 5), (10, 3), (10, 5), (20, 3), (20, 5), (40, 3), (40, 5)]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=list(datasets.values()),
        horizontal_spacing=0.15,
    )

    for col_idx, (dataset_key, dataset_name) in enumerate(datasets.items(), 1):
        std_csv = std_run / "results" / f"latency_{dataset_key}.csv"

        if not std_csv.exists():
            print(f"Warning: missing CSV for {dataset_key}")
            continue

        df_std = pd.read_csv(std_csv)

        # Filter to predict_proba mode, renode platform
        df_std = df_std[(df_std['benchmark_mode'] == 'predict_proba') & (df_std['platform'] == 'renode_nrf52840')]

        x_labels = []
        rf_cycles = []
        rf_labels = []
        gbt_std_cycles = []
        gbt_std_labels = []

        for n, d in configs:
            x_labels.append(f'n{n}_d{d}')

            # RF
            rf_row = df_std[(df_std['model_type'] == 'rf') &
                           (df_std['n_estimators'] == n) &
                           (df_std['max_depth'] == d)]
            if not rf_row.empty:
                rf_cycles.append(rf_row['avg_cycles'].values[0])
                acc = rf_row['accuracy'].values[0] * 100
                flash = rf_row['flash_bytes'].values[0] / 1024
                rf_labels.append(f'{acc:.0f}%<br>{flash:.1f}KB')
            else:
                rf_cycles.append(None)
                rf_labels.append('')

            # GBT standard
            gbt_row = df_std[(df_std['model_type'] == 'gbt') &
                            (df_std['n_estimators'] == n) &
                            (df_std['max_depth'] == d)]
            if not gbt_row.empty:
                gbt_std_cycles.append(gbt_row['avg_cycles'].values[0])
                acc = gbt_row['accuracy'].values[0] * 100
                flash = gbt_row['flash_bytes'].values[0] / 1024
                gbt_std_labels.append(f'{acc:.0f}%<br>{flash:.1f}KB')
            else:
                gbt_std_cycles.append(None)
                gbt_std_labels.append('')

        # Add traces
        fig.add_trace(go.Bar(
            name='RF',
            x=x_labels,
            y=rf_cycles,
            marker_color=MODEL_COLORS['rf'],
            text=rf_labels,
            textposition='outside',
            textfont=dict(size=8),
            showlegend=(col_idx == 1),
            legendgroup='rf',
        ), row=1, col=col_idx)

        fig.add_trace(go.Bar(
            name='GBT',
            x=x_labels,
            y=gbt_std_cycles,
            marker_color=MODEL_COLORS['gbt'],
            text=gbt_std_labels,
            textposition='outside',
            textfont=dict(size=8),
            showlegend=(col_idx == 1),
            legendgroup='gbt',
        ), row=1, col=col_idx)

    fig.update_yaxes(title_text="CPU Cycles (log scale)", type="log", col=1)
    fig.update_yaxes(type="log", col=2)
    fig.update_xaxes(title_text="Model Configuration", col=1)
    fig.update_xaxes(title_text="Model Configuration", col=2)

    fig.update_layout(
        title=dict(
            text='Binary Classification: Speed Comparison',
            x=0.5,
            font=dict(size=16),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        barmode="group",
        font=dict(size=12),
        margin=dict(t=80, b=80, l=60, r=40),
    )

    figures_dir.mkdir(exist_ok=True)
    filepath = figures_dir / 'binary_classification.png'
    fig.write_image(str(filepath), width=1200, height=500, scale=2)
    print(f"Saved: {filepath}")


def figure_multiclass_classification(results_dir: Path, figures_dir: Path):
    """Figure: Multi-class classification with RF vs GBT speed comparison."""
    runs_parent = _runs_dir(results_dir)

    std_run = runs_parent / LATENCY_STD_RUN

    datasets = {
        'iris': 'Iris (4 features, 3 classes)',
        'wine': 'Wine (13 features, 3 classes)',
        'embedded_synth': 'Embedded Synth (15 features, 3 classes)',
        'digits': 'Digits (64 features, 10 classes)',
    }
    configs = [(3, 3), (3, 5), (10, 3), (10, 5), (20, 3), (20, 5), (40, 3), (40, 5)]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(datasets.values()),
        horizontal_spacing=0.12,
        vertical_spacing=0.20,
    )

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for (row, col), (dataset_key, dataset_name) in zip(positions, datasets.items()):
        std_csv = std_run / "results" / f"latency_{dataset_key}.csv"

        if not std_csv.exists():
            print(f"Warning: missing CSV for {dataset_key}")
            continue

        df_std = pd.read_csv(std_csv)

        # Filter to predict_proba mode, renode platform
        df_std = df_std[(df_std['benchmark_mode'] == 'predict_proba') & (df_std['platform'] == 'renode_nrf52840')]

        x_labels = []
        rf_cycles = []
        rf_labels = []
        gbt_std_cycles = []
        gbt_std_labels = []

        for n, d in configs:
            x_labels.append(f'n{n}_d{d}')

            # RF
            rf_row = df_std[(df_std['model_type'] == 'rf') &
                           (df_std['n_estimators'] == n) &
                           (df_std['max_depth'] == d)]
            if not rf_row.empty:
                rf_cycles.append(rf_row['avg_cycles'].values[0])
                acc = rf_row['accuracy'].values[0] * 100
                flash = rf_row['flash_bytes'].values[0] / 1024
                rf_labels.append(f'{acc:.0f}%<br>{flash:.1f}KB')
            else:
                rf_cycles.append(None)
                rf_labels.append('')

            # GBT standard
            gbt_row = df_std[(df_std['model_type'] == 'gbt') &
                            (df_std['n_estimators'] == n) &
                            (df_std['max_depth'] == d)]
            if not gbt_row.empty:
                gbt_std_cycles.append(gbt_row['avg_cycles'].values[0])
                acc = gbt_row['accuracy'].values[0] * 100
                flash = gbt_row['flash_bytes'].values[0] / 1024
                gbt_std_labels.append(f'{acc:.0f}%<br>{flash:.1f}KB')
            else:
                gbt_std_cycles.append(None)
                gbt_std_labels.append('')

        # Add traces (only show legend in first subplot)
        show_legend = (row == 1 and col == 1)

        fig.add_trace(go.Bar(
            name='RF',
            x=x_labels,
            y=rf_cycles,
            marker_color=MODEL_COLORS['rf'],
            text=rf_labels,
            textposition='outside',
            textfont=dict(size=7),
            showlegend=show_legend,
            legendgroup='rf',
        ), row=row, col=col)

        fig.add_trace(go.Bar(
            name='GBT',
            x=x_labels,
            y=gbt_std_cycles,
            marker_color=MODEL_COLORS['gbt'],
            text=gbt_std_labels,
            textposition='outside',
            textfont=dict(size=7),
            showlegend=show_legend,
            legendgroup='gbt',
        ), row=row, col=col)

    # Update axes
    fig.update_yaxes(title_text="CPU Cycles (log scale)", type="log", row=1, col=1)
    fig.update_yaxes(title_text="CPU Cycles (log scale)", type="log", row=2, col=1)
    fig.update_yaxes(type="log", row=1, col=2)
    fig.update_yaxes(type="log", row=2, col=2)

    for r in [1, 2]:
        for c in [1, 2]:
            fig.update_xaxes(title_text="Model Configuration", row=r, col=c, tickangle=-45)

    fig.update_layout(
        title=dict(
            text='Multi-class Classification: Speed Comparison',
            x=0.5,
            font=dict(size=16),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        barmode="group",
        font=dict(size=12),
        margin=dict(t=80, b=100, l=60, r=40),
    )

    figures_dir.mkdir(exist_ok=True)
    filepath = figures_dir / 'multiclass_classification.png'
    fig.write_image(str(filepath), width=1200, height=900, scale=2)
    print(f"Saved: {filepath}")


def figure_binary_classification_accuracy_curves(results_dir: Path, figures_dir: Path):
    """Figure: Binary classification accuracy curves by model complexity.

    Shows how RF and GBT accuracy improve with model complexity
    (n_estimators and max_depth) for binary classification datasets.
    """
    runs_parent = _runs_dir(results_dir)

    std_run = runs_parent / LATENCY_STD_RUN

    if not (std_run / "results").exists():
        print("Skipping binary_classification_accuracy_curves: need standard run data")
        return

    datasets = {
        'breast_cancer': 'Breast Cancer',
        'sonar': 'Sonar',
    }
    configs = [(3, 3), (3, 5), (10, 3), (10, 5), (20, 3), (20, 5), (40, 3), (40, 5)]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=list(datasets.values()),
        horizontal_spacing=0.15,
    )

    for col_idx, (dataset_key, dataset_name) in enumerate(datasets.items(), 1):
        std_csv = std_run / "results" / f"latency_{dataset_key}.csv"

        if not std_csv.exists():
            print(f"Warning: missing CSV for {dataset_key}")
            continue

        df_std = pd.read_csv(std_csv)

        # Filter to predict_proba mode, renode platform
        df_std = df_std[(df_std['benchmark_mode'] == 'predict_proba') & (df_std['platform'] == 'renode_nrf52840')]

        # Collect accuracy data for each model type
        rf_accuracies = []
        gbt_accuracies = []
        x_labels = []

        for n, d in configs:
            x_labels.append(f'n{n}_d{d}')

            # RF
            rf_row = df_std[(df_std['model_type'] == 'rf') &
                           (df_std['n_estimators'] == n) &
                           (df_std['max_depth'] == d)]
            rf_accuracies.append(rf_row['accuracy'].values[0] * 100 if not rf_row.empty else None)

            # GBT
            gbt_row = df_std[(df_std['model_type'] == 'gbt') &
                            (df_std['n_estimators'] == n) &
                            (df_std['max_depth'] == d)]
            gbt_accuracies.append(gbt_row['accuracy'].values[0] * 100 if not gbt_row.empty else None)

        # Add line traces for each model type
        fig.add_trace(go.Scatter(
            name='RF',
            x=x_labels,
            y=rf_accuracies,
            mode='lines+markers',
            marker=dict(size=8, color=MODEL_COLORS['rf']),
            line=dict(color=MODEL_COLORS['rf'], width=2),
            showlegend=(col_idx == 1),
            legendgroup='rf',
            hovertemplate='<b>RF</b><br>%{x}<br>Accuracy: %{y:.1f}%<extra></extra>',
        ), row=1, col=col_idx)

        fig.add_trace(go.Scatter(
            name='GBT',
            x=x_labels,
            y=gbt_accuracies,
            mode='lines+markers',
            marker=dict(size=8, color=MODEL_COLORS['gbt']),
            line=dict(color=MODEL_COLORS['gbt'], width=2),
            showlegend=(col_idx == 1),
            legendgroup='gbt',
            hovertemplate='<b>GBT</b><br>%{x}<br>Accuracy: %{y:.1f}%<extra></extra>',
        ), row=1, col=col_idx)

    fig.update_yaxes(title_text="Accuracy (%)", col=1)
    fig.update_yaxes(col=2)
    fig.update_xaxes(title_text="Model Configuration", col=1, tickangle=-45)
    fig.update_xaxes(title_text="Model Configuration", col=2, tickangle=-45)

    fig.update_layout(
        title=dict(
            text='Binary Classification: Accuracy Score Comparison',
            x=0.5,
            font=dict(size=16),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        font=dict(size=12),
        margin=dict(t=80, b=100, l=60, r=40),
    )

    figures_dir.mkdir(exist_ok=True)
    filepath = figures_dir / 'binary_classification_accuracy_curves.png'
    fig.write_image(str(filepath), width=1200, height=500, scale=2)
    print(f"Saved: {filepath}")


def figure_binary_flash_vs_accuracy(results_dir: Path, figures_dir: Path):
    """Figure: Binary classification flash vs accuracy lines grouped by depth."""
    runs_parent = _runs_dir(results_dir)

    std_run = runs_parent / LATENCY_STD_RUN

    if not (std_run / "results").exists():
        print("Skipping binary_flash_vs_accuracy: need standard run data")
        return

    datasets = {
        'breast_cancer': 'Breast Cancer',
        'sonar': 'Sonar',
    }
    n_values = [3, 10, 20, 40]  # Full range

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=list(datasets.values()),
        horizontal_spacing=0.15,
    )

    for col_idx, (dataset_key, dataset_name) in enumerate(datasets.items(), 1):
        std_csv = std_run / "results" / f"latency_{dataset_key}.csv"

        if not std_csv.exists():
            print(f"Warning: missing CSV for {dataset_key}")
            continue

        df_std = pd.read_csv(std_csv)

        # Filter to predict_proba mode, renode platform
        df_std = df_std[(df_std['benchmark_mode'] == 'predict_proba') & (df_std['platform'] == 'renode_nrf52840')]

        # Create 4 lines: RF d=3, RF d=5, GBT d=3, GBT d=5
        for model_type, base_color, model_name in [
            ('rf', MODEL_COLORS['rf'], 'RF'),
            ('gbt', MODEL_COLORS['gbt'], 'GBT')
        ]:
            for depth, dash_style in [(3, 'solid'), (5, 'dash')]:
                flash_vals = []
                acc_vals = []

                for n in n_values:
                    row = df_std[(df_std['model_type'] == model_type) &
                                (df_std['n_estimators'] == n) &
                                (df_std['max_depth'] == depth)]
                    if not row.empty:
                        flash_vals.append(row['flash_bytes'].values[0] / 1024)
                        acc_vals.append(row['accuracy'].values[0] * 100)

                if not flash_vals:
                    continue

                line_name = f'{model_name} d={depth}'

                fig.add_trace(go.Scatter(
                    name=line_name,
                    x=flash_vals,
                    y=acc_vals,
                    mode='lines+markers',
                    line=dict(color=base_color, width=2, dash=dash_style),
                    marker=dict(size=8, color=base_color),
                    hovertemplate=f'<b>{line_name}</b><br>Flash: %{{x:.1f}} KB<br>Accuracy: %{{y:.1f}}%<extra></extra>',
                    showlegend=(col_idx == 1),
                    legendgroup=f'{model_type}_d{depth}',
                ), row=1, col=col_idx)

    fig.update_xaxes(title_text="Flash Size (KB)", col=1)
    fig.update_xaxes(title_text="Flash Size (KB)", col=2)
    fig.update_yaxes(title_text="Accuracy (%)", col=1)
    fig.update_yaxes(col=2)

    fig.update_layout(
        title=dict(
            text='Binary Classification: Flash vs Accuracy Trade-off',
            x=0.5,
            font=dict(size=16),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        font=dict(size=12),
        margin=dict(t=80, b=60, l=60, r=40),
    )

    figures_dir.mkdir(exist_ok=True)
    filepath = figures_dir / 'binary_flash_vs_accuracy.png'
    fig.write_image(str(filepath), width=1200, height=500, scale=2)
    print(f"Saved: {filepath}")


def figure_multiclass_flash_vs_accuracy(results_dir: Path, figures_dir: Path):
    """Figure: Multi-class classification flash vs accuracy lines grouped by depth."""
    runs_parent = _runs_dir(results_dir)

    std_run = runs_parent / LATENCY_STD_RUN

    if not (std_run / "results").exists():
        print("Skipping multiclass_flash_vs_accuracy: need standard run data")
        return

    datasets = {
        'iris': 'Iris',
        'wine': 'Wine',
        'embedded_synth': 'Embedded Synth',
        'digits': 'Digits',
    }
    n_values = [3, 10, 20, 40]  # Full range

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(datasets.values()),
        horizontal_spacing=0.12,
        vertical_spacing=0.15,
    )

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for (row, col), (dataset_key, dataset_name) in zip(positions, datasets.items()):
        std_csv = std_run / "results" / f"latency_{dataset_key}.csv"

        if not std_csv.exists():
            print(f"Warning: missing CSV for {dataset_key}")
            continue

        df_std = pd.read_csv(std_csv)

        # Filter to predict_proba mode, renode platform
        df_std = df_std[(df_std['benchmark_mode'] == 'predict_proba') & (df_std['platform'] == 'renode_nrf52840')]

        # Create 4 lines: RF d=3, RF d=5, GBT d=3, GBT d=5
        show_legend = (row == 1 and col == 1)
        for model_type, base_color, model_name in [
            ('rf', MODEL_COLORS['rf'], 'RF'),
            ('gbt', MODEL_COLORS['gbt'], 'GBT')
        ]:
            for depth, dash_style in [(3, 'solid'), (5, 'dash')]:
                flash_vals = []
                acc_vals = []

                for n in n_values:
                    row_data = df_std[(df_std['model_type'] == model_type) &
                                     (df_std['n_estimators'] == n) &
                                     (df_std['max_depth'] == depth)]
                    if not row_data.empty:
                        flash_vals.append(row_data['flash_bytes'].values[0] / 1024)
                        acc_vals.append(row_data['accuracy'].values[0] * 100)

                if not flash_vals:
                    continue

                line_name = f'{model_name} d={depth}'

                fig.add_trace(go.Scatter(
                    name=line_name,
                    x=flash_vals,
                    y=acc_vals,
                    mode='lines+markers',
                    line=dict(color=base_color, width=2, dash=dash_style),
                    marker=dict(size=8, color=base_color),
                    hovertemplate=f'<b>{line_name}</b><br>Flash: %{{x:.1f}} KB<br>Accuracy: %{{y:.1f}}%<extra></extra>',
                    showlegend=show_legend,
                    legendgroup=f'{model_type}_d{depth}',
                ), row=row, col=col)

    fig.update_xaxes(title_text="Flash Size (KB)", row=1, col=1)
    fig.update_xaxes(title_text="Flash Size (KB)", row=1, col=2)
    fig.update_xaxes(title_text="Flash Size (KB)", row=2, col=1)
    fig.update_xaxes(title_text="Flash Size (KB)", row=2, col=2)
    fig.update_yaxes(title_text="Accuracy (%)", row=1, col=1)
    fig.update_yaxes(title_text="Accuracy (%)", row=2, col=1)
    fig.update_yaxes(row=1, col=2)
    fig.update_yaxes(row=2, col=2)

    fig.update_layout(
        title=dict(
            text='Multi-class Classification: Flash vs Accuracy Trade-off',
            x=0.5,
            font=dict(size=16),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        font=dict(size=12),
        margin=dict(t=80, b=60, l=60, r=40),
    )

    figures_dir.mkdir(exist_ok=True)
    filepath = figures_dir / 'multiclass_flash_vs_accuracy.png'
    fig.write_image(str(filepath), width=1200, height=900, scale=2)
    print(f"Saved: {filepath}")


def figure_binary_calibration_curves(results_dir: Path, figures_dir: Path):
    """Figure: Binary classification calibration quality curves grouped by depth."""
    runs_parent = _runs_dir(results_dir)

    calib_run = runs_parent / CALIB_RUN_NAME

    if not (calib_run / "results").exists():
        print("Skipping binary_calibration_curves: need calibration run data")
        return

    calib_csv = calib_run / "results" / "probability_calibration.csv"
    if not calib_csv.exists():
        print("Skipping binary_calibration_curves: no calibration data")
        return

    df = pd.read_csv(calib_csv)

    # Filter to binary datasets
    datasets = {
        'breast_cancer': 'Breast Cancer',
        'sonar': 'Sonar',
    }

    df = df[df['dataset'].isin(datasets.keys())]
    n_values = [3, 10, 20, 40]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=list(datasets.values()),
        horizontal_spacing=0.15,
    )

    for col_idx, (dataset_key, dataset_name) in enumerate(datasets.items(), 1):
        df_dataset = df[df['dataset'] == dataset_key]

        for model_type, base_color, model_name in [
            ('rf', MODEL_COLORS['rf'], 'RF'),
            ('gbt', MODEL_COLORS['gbt'], 'GBT')
        ]:
            for depth, dash_style in [(3, 'solid'), (5, 'dash')]:
                n_vals = []
                brier_vals = []

                for n in n_values:
                    row = df_dataset[(df_dataset['model_type'] == model_type) &
                                    (df_dataset['n_estimators'] == n) &
                                    (df_dataset['max_depth'] == depth)]
                    if not row.empty:
                        n_vals.append(n)
                        brier_vals.append(row['brier_score'].values[0])

                if not n_vals:
                    continue

                line_name = f'{model_name} d={depth}'

                fig.add_trace(go.Scatter(
                    name=line_name,
                    x=n_vals,
                    y=brier_vals,
                    mode='lines+markers',
                    line=dict(color=base_color, width=2, dash=dash_style),
                    marker=dict(size=8, color=base_color),
                    hovertemplate=f'<b>{line_name}</b><br>n=%{{x}}<br>Brier Score: %{{y:.3f}}<extra></extra>',
                    showlegend=(col_idx == 1),
                    legendgroup=f'{model_type}_d{depth}',
                ), row=1, col=col_idx)

    fig.update_xaxes(title_text="Number of Trees (n_estimators)", col=1)
    fig.update_xaxes(title_text="Number of Trees (n_estimators)", col=2)
    fig.update_yaxes(title_text="Brier Score (lower is better)", col=1)
    fig.update_yaxes(col=2)

    fig.update_layout(
        title=dict(
            text='Binary Classification: Calibration Quality vs Model Complexity',
            x=0.5,
            font=dict(size=16),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        font=dict(size=12),
        margin=dict(t=80, b=60, l=60, r=40),
    )

    figures_dir.mkdir(exist_ok=True)
    filepath = figures_dir / 'binary_calibration_curves.png'
    fig.write_image(str(filepath), width=1200, height=500, scale=2)
    print(f"Saved: {filepath}")


def figure_multiclass_calibration_curves(results_dir: Path, figures_dir: Path):
    """Figure: Multi-class classification calibration quality curves grouped by depth."""
    runs_parent = _runs_dir(results_dir)

    calib_run = runs_parent / CALIB_RUN_NAME

    if not (calib_run / "results").exists():
        print("Skipping multiclass_calibration_curves: need calibration run data")
        return

    calib_csv = calib_run / "results" / "probability_calibration.csv"
    if not calib_csv.exists():
        print("Skipping multiclass_calibration_curves: no calibration data")
        return

    df = pd.read_csv(calib_csv)

    # Filter to multiclass datasets
    datasets = {
        'iris': 'Iris',
        'wine': 'Wine',
        'embedded_synth': 'Embedded Synth',
        'digits': 'Digits',
    }

    df = df[df['dataset'].isin(datasets.keys())]
    n_values = [3, 10, 20, 40]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=list(datasets.values()),
        horizontal_spacing=0.12,
        vertical_spacing=0.15,
    )

    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
    for (row, col), (dataset_key, dataset_name) in zip(positions, datasets.items()):
        df_dataset = df[df['dataset'] == dataset_key]

        show_legend = (row == 1 and col == 1)
        for model_type, base_color, model_name in [
            ('rf', MODEL_COLORS['rf'], 'RF'),
            ('gbt', MODEL_COLORS['gbt'], 'GBT')
        ]:
            for depth, dash_style in [(3, 'solid'), (5, 'dash')]:
                n_vals = []
                brier_vals = []

                for n in n_values:
                    row_data = df_dataset[(df_dataset['model_type'] == model_type) &
                                         (df_dataset['n_estimators'] == n) &
                                         (df_dataset['max_depth'] == depth)]
                    if not row_data.empty:
                        n_vals.append(n)
                        brier_vals.append(row_data['brier_score'].values[0])

                if not n_vals:
                    continue

                line_name = f'{model_name} d={depth}'

                fig.add_trace(go.Scatter(
                    name=line_name,
                    x=n_vals,
                    y=brier_vals,
                    mode='lines+markers',
                    line=dict(color=base_color, width=2, dash=dash_style),
                    marker=dict(size=8, color=base_color),
                    hovertemplate=f'<b>{line_name}</b><br>n=%{{x}}<br>Brier Score: %{{y:.3f}}<extra></extra>',
                    showlegend=show_legend,
                    legendgroup=f'{model_type}_d{depth}',
                ), row=row, col=col)

    fig.update_xaxes(title_text="Number of Trees (n_estimators)", row=1, col=1)
    fig.update_xaxes(title_text="Number of Trees (n_estimators)", row=1, col=2)
    fig.update_xaxes(title_text="Number of Trees (n_estimators)", row=2, col=1)
    fig.update_xaxes(title_text="Number of Trees (n_estimators)", row=2, col=2)
    fig.update_yaxes(title_text="Brier Score (lower is better)", row=1, col=1)
    fig.update_yaxes(title_text="Brier Score (lower is better)", row=2, col=1)
    fig.update_yaxes(row=1, col=2)
    fig.update_yaxes(row=2, col=2)

    fig.update_layout(
        title=dict(
            text='Multi-class Classification: Calibration Quality vs Model Complexity',
            x=0.5,
            font=dict(size=16),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        font=dict(size=12),
        margin=dict(t=80, b=60, l=60, r=40),
    )

    figures_dir.mkdir(exist_ok=True)
    filepath = figures_dir / 'multiclass_calibration_curves.png'
    fig.write_image(str(filepath), width=1200, height=900, scale=2)
    print(f"Saved: {filepath}")


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []

    try:
        import plotly
        print(f"plotly: {plotly.__version__}")
    except ImportError:
        missing.append("plotly")

    try:
        import kaleido  # noqa: F401
        print("kaleido: installed")
    except ImportError:
        missing.append("kaleido")

    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False

    print("\nAll dependencies installed.")
    return True


def generate_all_figures(run_dir: Path | None = None):
    """Generate all figures from run results.

    Args:
        run_dir: Path to run directory (with results/ subdirectory).
                 If None, uses default RESULTS_DIR.
    """
    if run_dir is not None:
        results_dir = run_dir / "results"
        figures_dir = run_dir / "figures"
    else:
        results_dir = RESULTS_DIR
        figures_dir = FIGURES_DIR

    print(f"Results directory: {results_dir}")
    print(f"Output directory: {figures_dir}")
    print()

    # Legacy figures (from mcu_inference_timing.csv)
    if (results_dir / "mcu_inference_timing.csv").exists():
        figure_platform_comparison()
        figure_cycles_vs_accuracy()
        figure_gbt_vs_rf()
        figure_method_comparison()
        figure_timing_breakdown()
        figure_fpu_comparison()

    # Find full sweep data (for benchmarks not in latency-focused runs)
    if run_dir is not None:
        runs_parent = run_dir.parent  # run_dir = runs/<timestamp>, parent = runs/
    else:
        runs_parent = SCRIPT_DIR / "runs"
    full_sweep_results = runs_parent / FULL_SWEEP_RUN / "results"
    full_src = full_sweep_results if full_sweep_results.exists() else results_dir

    # New benchmark figures (use full sweep data when available)
    figure_sample_efficiency_classification(full_src, figures_dir)
    figure_sample_efficiency_regression(full_src, figures_dir)
    figure_pareto_efficiency(full_src, figures_dir)
    figure_size_constrained(full_src, figures_dir)
    figure_regression_accuracy(full_src, figures_dir)

    # High-priority figures for README
    figure_calibration_comparison(results_dir, figures_dir)
    figure_binary_classification(results_dir, figures_dir)
    figure_binary_classification_accuracy_curves(results_dir, figures_dir)
    figure_binary_flash_vs_accuracy(results_dir, figures_dir)
    figure_binary_calibration_curves(results_dir, figures_dir)
    figure_multiclass_classification(results_dir, figures_dir)
    figure_multiclass_flash_vs_accuracy(results_dir, figures_dir)
    figure_multiclass_calibration_curves(results_dir, figures_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate MCU benchmark figures")
    parser.add_argument("--check", action="store_true", help="Check dependencies only")
    parser.add_argument("--input", "-i", type=str, default="mcu_inference_timing.csv",
                        help="Input CSV filename")
    parser.add_argument("run_dir", nargs="?", type=Path,
                        help="Run directory (optional, uses default results/ if not specified)")
    args = parser.parse_args()

    if args.check:
        sys.exit(0 if check_dependencies() else 1)

    print("Generating figures...")

    generate_all_figures(args.run_dir)

    print("\nDone!")


if __name__ == "__main__":
    main()
