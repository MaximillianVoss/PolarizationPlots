# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from tkinter import TclError, ttk


LIGHT_THEME = "light"
DARK_THEME = "dark"
DEFAULT_THEME_NAME = LIGHT_THEME
MIN_TEXT_CONTRAST = 4.5


@dataclass(frozen=True)
class AppTheme:
    name: str
    display_name: str
    background: str
    surface: str
    panel: str
    text: str
    muted: str
    border: str
    accent: str
    accent_hover: str
    on_accent: str
    input_background: str
    input_text: str
    selection_background: str
    selection_text: str
    plot_background: str
    plot_grid: str
    error: str
    error_background: str
    warning: str
    warning_background: str
    success: str
    success_background: str
    disabled: str
    tooltip_background: str
    tooltip_text: str


THEMES: dict[str, AppTheme] = {
    LIGHT_THEME: AppTheme(
        name=LIGHT_THEME,
        display_name="Светлая",
        background="#f4f6f8",
        surface="#ffffff",
        panel="#f8fafc",
        text="#18202a",
        muted="#59636e",
        border="#d6dbe3",
        accent="#2563eb",
        accent_hover="#1d4ed8",
        on_accent="#ffffff",
        input_background="#ffffff",
        input_text="#111827",
        selection_background="#bfdbfe",
        selection_text="#111827",
        plot_background="#ffffff",
        plot_grid="#d8dde5",
        error="#b91c1c",
        error_background="#fef2f2",
        warning="#b45309",
        warning_background="#fff7ed",
        success="#15803d",
        success_background="#ecfdf5",
        disabled="#9ca3af",
        tooltip_background="#fffbe6",
        tooltip_text="#111827",
    ),
    DARK_THEME: AppTheme(
        name=DARK_THEME,
        display_name="Тёмная",
        background="#111827",
        surface="#172033",
        panel="#1f2937",
        text="#e5edf7",
        muted="#b7c0cc",
        border="#384456",
        accent="#60a5fa",
        accent_hover="#93c5fd",
        on_accent="#0b1220",
        input_background="#0f172a",
        input_text="#f8fafc",
        selection_background="#1d4ed8",
        selection_text="#ffffff",
        plot_background="#111827",
        plot_grid="#344154",
        error="#f87171",
        error_background="#351b1f",
        warning="#fbbf24",
        warning_background="#2f2412",
        success="#4ade80",
        success_background="#132c1d",
        disabled="#6b7280",
        tooltip_background="#243244",
        tooltip_text="#f8fafc",
    ),
}


def apply_ttk_theme(style: ttk.Style, theme: AppTheme) -> None:
    try:
        style.theme_use("clam")
    except TclError:
        pass

    style.configure(
        ".",
        background=theme.background,
        foreground=theme.text,
        fieldbackground=theme.input_background,
        selectbackground=theme.selection_background,
        selectforeground=theme.selection_text,
        troughcolor=theme.border,
        bordercolor=theme.border,
        lightcolor=theme.surface,
        darkcolor=theme.border,
    )
    style.configure("TFrame", background=theme.background)
    style.configure("Header.TFrame", background=theme.surface)
    style.configure("Toolbar.TFrame", background=theme.surface)
    style.configure("Content.TFrame", background=theme.background)
    style.configure("Panel.TFrame", background=theme.surface)
    style.configure("Card.TFrame", background=theme.surface, bordercolor=theme.border, relief="solid")
    style.configure("TLabel", background=theme.background, foreground=theme.text)
    style.configure("Muted.TLabel", background=theme.background, foreground=theme.muted)
    style.configure("Error.TLabel", background=theme.background, foreground=theme.error)
    style.configure("Success.TLabel", background=theme.background, foreground=theme.success)
    style.configure("Title.TLabel", background=theme.surface, foreground=theme.text, font=("Segoe UI", 13, "bold"))
    style.configure("HeaderMuted.TLabel", background=theme.surface, foreground=theme.muted)
    style.configure("CardTitle.TLabel", background=theme.surface, foreground=theme.text, font=("Segoe UI", 9, "bold"))
    style.configure("CardText.TLabel", background=theme.surface, foreground=theme.text)
    style.configure("CardMuted.TLabel", background=theme.surface, foreground=theme.muted)
    style.configure("MetricValue.TLabel", background=theme.surface, foreground=theme.accent, font=("Segoe UI", 9, "bold"))
    style.configure("Status.TLabel", background=theme.surface, foreground=theme.success, font=("Segoe UI", 9, "bold"))
    style.configure("TLabelframe", background=theme.background, foreground=theme.text, bordercolor=theme.border)
    style.configure("TLabelframe.Label", background=theme.background, foreground=theme.text)
    style.configure("TButton", background=theme.surface, foreground=theme.text, bordercolor=theme.border, padding=(8, 4))
    style.map(
        "TButton",
        background=[("active", theme.panel), ("pressed", theme.border), ("disabled", theme.panel)],
        foreground=[("disabled", theme.disabled)],
    )
    style.configure(
        "Accent.TButton",
        background=theme.accent,
        foreground=theme.on_accent,
        bordercolor=theme.accent,
        padding=(14, 8),
        font=("Segoe UI", 9, "bold"),
    )
    style.map(
        "Accent.TButton",
        background=[("active", theme.accent_hover), ("pressed", theme.accent_hover), ("disabled", theme.panel)],
        foreground=[("disabled", theme.disabled)],
    )
    style.configure(
        "Toolbar.TButton",
        background=theme.surface,
        foreground=theme.text,
        bordercolor=theme.border,
        padding=(12, 8),
        font=("Segoe UI", 9),
    )
    style.map(
        "Toolbar.TButton",
        background=[("active", theme.panel), ("pressed", theme.border), ("disabled", theme.panel)],
        foreground=[("disabled", theme.disabled)],
    )
    style.configure(
        "Nav.TButton",
        background=theme.surface,
        foreground=theme.muted,
        bordercolor=theme.surface,
        padding=(12, 9),
        font=("Segoe UI", 10),
    )
    style.map(
        "Nav.TButton",
        background=[("active", theme.panel), ("pressed", theme.panel), ("disabled", theme.surface)],
        foreground=[("active", theme.text), ("disabled", theme.disabled)],
    )
    style.configure(
        "NavActive.TButton",
        background=theme.panel,
        foreground=theme.accent,
        bordercolor=theme.accent,
        padding=(12, 9),
        font=("Segoe UI", 10, "bold"),
    )
    style.map(
        "NavActive.TButton",
        background=[("active", theme.panel), ("pressed", theme.panel), ("disabled", theme.surface)],
        foreground=[("active", theme.accent), ("disabled", theme.disabled)],
    )
    style.configure("TCheckbutton", background=theme.background, foreground=theme.text)
    style.configure("TRadiobutton", background=theme.background, foreground=theme.text)
    style.map(
        "TCheckbutton",
        background=[("active", theme.background), ("disabled", theme.background)],
        foreground=[("disabled", theme.disabled)],
    )
    style.map(
        "TRadiobutton",
        background=[("active", theme.background), ("disabled", theme.background)],
        foreground=[("disabled", theme.disabled)],
    )
    style.configure(
        "TEntry",
        fieldbackground=theme.input_background,
        foreground=theme.input_text,
        insertcolor=theme.input_text,
        bordercolor=theme.border,
        lightcolor=theme.border,
        darkcolor=theme.border,
    )
    style.map(
        "TEntry",
        fieldbackground=[("disabled", theme.panel), ("readonly", theme.input_background)],
        foreground=[("disabled", theme.disabled)],
    )
    style.configure(
        "TCombobox",
        fieldbackground=theme.input_background,
        background=theme.input_background,
        foreground=theme.input_text,
        arrowcolor=theme.text,
        bordercolor=theme.border,
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", theme.input_background), ("disabled", theme.panel)],
        foreground=[("readonly", theme.input_text), ("disabled", theme.disabled)],
        selectbackground=[("readonly", theme.input_background)],
        selectforeground=[("readonly", theme.input_text)],
    )
    style.configure("TNotebook", background=theme.background, bordercolor=theme.border)
    style.configure("Shell.TNotebook", background=theme.background, borderwidth=0)
    style.configure(
        "TNotebook.Tab",
        background=theme.panel,
        foreground=theme.muted,
        bordercolor=theme.border,
        padding=(10, 5),
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", theme.surface), ("active", theme.surface)],
        foreground=[("selected", theme.accent), ("active", theme.text)],
    )
    style.configure("Horizontal.TScale", background=theme.background, troughcolor=theme.border)
    style.configure(
        "Rmin.Treeview",
        background=theme.surface,
        fieldbackground=theme.surface,
        foreground=theme.text,
        bordercolor=theme.border,
        rowheight=28,
    )
    style.configure(
        "Rmin.Treeview.Heading",
        background=theme.panel,
        foreground=theme.text,
        bordercolor=theme.border,
        font=("Segoe UI", 9, "bold"),
    )
    style.map("Rmin.Treeview", background=[("selected", theme.selection_background)], foreground=[("selected", theme.selection_text)])
    style.configure("Vertical.TScrollbar", background=theme.panel, troughcolor=theme.background, arrowcolor=theme.text)
    style.configure("Horizontal.TScrollbar", background=theme.panel, troughcolor=theme.background, arrowcolor=theme.text)
    style.configure("TPanedwindow", background=theme.background)
    style.configure("TSeparator", background=theme.border)
    try:
        style.layout("Shell.TNotebook.Tab", [])
    except TclError:
        pass


def apply_matplotlib_theme(figure, theme: AppTheme) -> None:
    if figure is None:
        return
    figure.set_facecolor(theme.background)
    for axis in figure.axes:
        axis.set_facecolor(theme.plot_background)
        axis.title.set_color(theme.text)
        axis.xaxis.label.set_color(theme.text)
        axis.yaxis.label.set_color(theme.text)
        axis.tick_params(axis="both", which="both", colors=theme.text)
        if hasattr(axis, "zaxis"):
            axis.zaxis.label.set_color(theme.text)
            axis.tick_params(axis="z", which="both", colors=theme.text)
        for spine in axis.spines.values():
            spine.set_color(theme.border)
        axis.grid(True, color=theme.plot_grid, alpha=0.85, linewidth=0.8)
        _style_axis_texts(axis, theme)
        _style_axis_legend(axis, theme)
        _style_3d_panes(axis, theme)


def _style_axis_texts(axis, theme: AppTheme) -> None:
    for text in axis.texts:
        current = str(text.get_color()).lower()
        if current in {"black", "#000000", "#111827", "#18202a", "#666"}:
            text.set_color(theme.text)
        bbox = text.get_bbox_patch()
        if bbox is None:
            continue
        if current in {"#991b1b", "#b91c1c", "#f87171"}:
            bbox.set_facecolor(theme.error_background)
            bbox.set_edgecolor(theme.error)
        else:
            bbox.set_facecolor(theme.warning_background)
            bbox.set_edgecolor(theme.warning)


def _style_axis_legend(axis, theme: AppTheme) -> None:
    legend = axis.get_legend()
    if legend is None:
        return
    legend.get_frame().set_facecolor(theme.surface)
    legend.get_frame().set_edgecolor(theme.border)
    legend.get_frame().set_alpha(0.92)
    for text in legend.get_texts():
        text.set_color(theme.text)


def _style_3d_panes(axis, theme: AppTheme) -> None:
    for pane_axis_name in ("xaxis", "yaxis", "zaxis"):
        pane_axis = getattr(axis, pane_axis_name, None)
        if pane_axis is None or not hasattr(pane_axis, "pane"):
            continue
        pane_axis.pane.set_facecolor(theme.plot_background)
        pane_axis.pane.set_edgecolor(theme.border)


def contrast_ratio(first: str, second: str) -> float:
    light = max(_relative_luminance(first), _relative_luminance(second))
    dark = min(_relative_luminance(first), _relative_luminance(second))
    return (light + 0.05) / (dark + 0.05)


def _relative_luminance(color: str) -> float:
    red, green, blue = _hex_to_rgb(color)
    linear = [_linearize(channel / 255.0) for channel in (red, green, blue)]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _linearize(value: float) -> float:
    if value <= 0.03928:
        return value / 12.92
    return ((value + 0.055) / 1.055) ** 2.4


def _hex_to_rgb(color: str) -> tuple[int, int, int]:
    normalized = color.strip().lstrip("#")
    if len(normalized) != 6:
        raise ValueError(f"Unsupported color format: {color}")
    return int(normalized[0:2], 16), int(normalized[2:4], 16), int(normalized[4:6], 16)


__all__ = [
    "AppTheme",
    "DARK_THEME",
    "DEFAULT_THEME_NAME",
    "LIGHT_THEME",
    "MIN_TEXT_CONTRAST",
    "THEMES",
    "apply_matplotlib_theme",
    "apply_ttk_theme",
    "contrast_ratio",
]
