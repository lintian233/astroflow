from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator

from .plots import AXIS_LABEL_FONTSIZE, AXIS_TICK_FONTSIZE, INFO_FONTSIZE, LEGEND_FONTSIZE


@dataclass
class DmPlotPayload:
    data: np.ndarray
    time_axis: np.ndarray
    dm_axis: np.ndarray
    vmin: float
    vmax: float
    dm: float
    toa: float


@dataclass
class SpectrumPlotPayload:
    image: np.ndarray
    extent: list[float]
    vmin: float
    vmax: float
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    time_x: np.ndarray
    time_y: np.ndarray
    freq_x: np.ndarray
    freq_y: np.ndarray
    freq_xlim: tuple[float, float]
    time_info: str
    info_text: str
    toa: Optional[float]
    time_ylabel: str = "Int. Power"
    ylabel: str = "Frequency (MHz)"
    xlabel: str = "Time (s)"
    freq_title: str = "Freq. Int. Power"
    freq_color: str = "black"
    interpolation: str = "nearest"


@dataclass
class CandidatePlotPayload:
    dm_plot: DmPlotPayload
    spectrum_plot: Optional[SpectrumPlotPayload]
    title: str
    title_y: float
    imgname: str
    dm_imgname: Optional[str]
    save_path: str
    savetype: str
    snr: float
    pulse_width_ms: float
    ref_toa: float


class CandidatePlotSession:
    """Reusable Matplotlib artists for all candidates from one input file."""

    def __init__(self, dpi: int = 150, onlyspec: bool = False):
        self.dpi = dpi
        self.onlyspec = onlyspec
        if onlyspec:
            self.dm_fig = plt.figure(figsize=(10, 10), dpi=dpi)
            self.fig = plt.figure(figsize=(10, 10), dpi=dpi)
            dm_axes = _single_dm_axes(self.dm_fig)
            spec_axes = _single_spec_axes(self.fig)
        else:
            self.dm_fig = None
            self.fig = plt.figure(figsize=(22, 10), dpi=dpi)
            dm_axes, spec_axes = _combined_axes(self.fig)

        self._init_dm_artists(dm_axes)
        self._init_spectrum_artists(spec_axes)

    def render(self, payload: CandidatePlotPayload) -> None:
        self._update_dm(payload.dm_plot)
        if payload.spectrum_plot is None:
            self._clear_spectrum()
        else:
            self._update_spectrum(payload.spectrum_plot)
        self.fig.suptitle(payload.title, fontsize=22, y=payload.title_y)

    def close(self) -> None:
        if self.dm_fig is not None:
            plt.close(self.dm_fig)
        plt.close(self.fig)

    def _init_dm_artists(self, axes: dict[str, object]) -> None:
        self.ax_dm_time = axes["time"]
        self.ax_dm_main = axes["main"]
        self.ax_dm_side = axes["side"]

        self.dm_image = self.ax_dm_main.imshow(
            np.zeros((1, 1), dtype=np.float32),
            aspect="auto",
            origin="lower",
            cmap="viridis",
            extent=[0, 1, 0, 1],
        )
        self.ax_dm_main.set_xlabel("Time (s)", fontsize=AXIS_LABEL_FONTSIZE, labelpad=8)
        self.ax_dm_main.set_ylabel("DM (pc cm$^{-3}$)", fontsize=AXIS_LABEL_FONTSIZE, labelpad=10)
        self.ax_dm_main.tick_params(axis="x", labelsize=AXIS_TICK_FONTSIZE)
        self.ax_dm_main.tick_params(axis="y", labelsize=AXIS_TICK_FONTSIZE)
        self.ax_dm_main.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="upper"))

        self.dm_ellipse = mpatches.Ellipse(
            (0, 0),
            width=0,
            height=0,
            fill=False,
            linestyle="--",
            linewidth=2,
            edgecolor="white",
            alpha=0.7,
        )
        self.ax_dm_main.add_patch(self.dm_ellipse)

        (self.dm_side_line,) = self.ax_dm_side.plot([], [], lw=1.5, color="darkblue")
        self.ax_dm_side.tick_params(axis="y", labelleft=False, labelsize=AXIS_TICK_FONTSIZE)
        self.ax_dm_side.tick_params(axis="x", labelsize=AXIS_TICK_FONTSIZE)
        self.ax_dm_side.set_title("DM Int.", fontsize=AXIS_LABEL_FONTSIZE, pad=6)
        self.ax_dm_side.grid(alpha=0.3)

        (self.dm_time_line,) = self.ax_dm_time.plot([], [], lw=1.5, color="darkred")
        self.ax_dm_time.tick_params(axis="x", labelbottom=False, labelsize=AXIS_TICK_FONTSIZE)
        self.ax_dm_time.tick_params(axis="y", labelsize=AXIS_TICK_FONTSIZE)
        self.ax_dm_time.set_ylabel("T Int.", fontsize=AXIS_LABEL_FONTSIZE, labelpad=6)
        self.ax_dm_time.grid(alpha=0.3)
        self.dm_info_text = self.ax_dm_time.text(
            0.02,
            0.95,
            "",
            transform=self.ax_dm_time.transAxes,
            fontsize=INFO_FONTSIZE,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    def _init_spectrum_artists(self, axes: dict[str, object]) -> None:
        self.ax_spec_time = axes["time"]
        self.ax_spec_main = axes["main"]
        self.ax_spec_side = axes["side"]

        (self.spec_time_line,) = self.ax_spec_time.plot([], [], "-", color="black", linewidth=1, alpha=0.9)
        self.spec_toa_line = self.ax_spec_time.axvline(
            0,
            color="blue",
            linestyle="--",
            linewidth=1,
            alpha=0.8,
            label="TOA",
        )
        self.spec_toa_line.set_visible(False)
        self.spec_time_info_text = self.ax_spec_time.text(
            0.02,
            0.96,
            "",
            transform=self.ax_spec_time.transAxes,
            fontsize=INFO_FONTSIZE,
            verticalalignment="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )
        self.ax_spec_time.set_ylabel("Int. Power", fontsize=AXIS_LABEL_FONTSIZE)
        self.ax_spec_time.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        self.ax_spec_time.tick_params(axis="y", labelsize=AXIS_TICK_FONTSIZE)
        self.ax_spec_time.grid(True, alpha=0.3)
        self.spec_legend = self.ax_spec_time.legend(fontsize=LEGEND_FONTSIZE, loc="upper right")

        self.spec_image = self.ax_spec_main.imshow(
            np.zeros((1, 1), dtype=np.float32),
            aspect="auto",
            origin="lower",
            cmap="viridis",
            extent=[0, 1, 0, 1],
            interpolation="nearest",
        )
        self.ax_spec_main.set_ylabel("Frequency (MHz)", fontsize=AXIS_LABEL_FONTSIZE)
        self.ax_spec_main.set_xlabel("Time (s)", fontsize=AXIS_LABEL_FONTSIZE, labelpad=8)
        self.ax_spec_main.tick_params(axis="x", labelsize=AXIS_TICK_FONTSIZE)
        self.ax_spec_main.tick_params(axis="y", labelsize=AXIS_TICK_FONTSIZE)
        self.ax_spec_main.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="upper"))

        (self.spec_side_line,) = self.ax_spec_side.plot([], [], "-", color="black", linewidth=1, alpha=0.8)
        self.ax_spec_side.tick_params(axis="y", which="both", left=False, labelleft=False)
        self.ax_spec_side.grid(True, alpha=0.3)
        self.ax_spec_side.set_title("Freq. Int. Power", fontsize=AXIS_LABEL_FONTSIZE, pad=6)
        self.ax_spec_side.tick_params(axis="x", which="both", bottom=True, top=False, labelbottom=True)

        self.ax_spec_info = axes["info"]
        self.ax_spec_info.axis("off")
        self.spec_info_text = self.ax_spec_info.text(
            0.98,
            0.98,
            "",
            transform=self.ax_spec_info.transAxes,
            fontsize=INFO_FONTSIZE,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    def _update_dm(self, payload: DmPlotPayload) -> None:
        self.dm_image.set_data(payload.data)
        self.dm_image.set_extent([payload.time_axis[0], payload.time_axis[-1], payload.dm_axis[0], payload.dm_axis[-1]])
        self.dm_image.set_clim(payload.vmin, payload.vmax)

        time_range = payload.time_axis[-1] - payload.time_axis[0]
        dm_range = payload.dm_axis[-1] - payload.dm_axis[0]
        self.dm_ellipse.center = (payload.toa, payload.dm)
        self.dm_ellipse.width = 2 * time_range * 0.05
        self.dm_ellipse.height = 2 * dm_range * 0.07

        dm_sum = np.max(payload.data, axis=1)
        time_sum = np.max(payload.data, axis=0)
        self.dm_side_line.set_data(dm_sum, payload.dm_axis)
        self.dm_time_line.set_data(payload.time_axis, time_sum)

        self.ax_dm_main.set_xlim(payload.time_axis[0], payload.time_axis[-1])
        self.ax_dm_main.set_ylim(payload.dm_axis[0], payload.dm_axis[-1])
        self.ax_dm_side.set_ylim(payload.dm_axis[0], payload.dm_axis[-1])
        self.ax_dm_time.set_xlim(payload.time_axis[0], payload.time_axis[-1])
        _autoscale_axis(self.ax_dm_side, scale_x=True, scale_y=False)
        _autoscale_axis(self.ax_dm_time, scale_x=False, scale_y=True)

        self.dm_info_text.set_text(f"DM: {payload.dm:.2f} pc $cm^{{-3}}$ \n TOA: {payload.toa:.3f}s")

    def _update_spectrum(self, payload: SpectrumPlotPayload) -> None:
        self.spec_image.set_data(payload.image.T)
        self.spec_image.set_extent(payload.extent)
        self.spec_image.set_clim(payload.vmin, payload.vmax)
        self.spec_image.set_interpolation(payload.interpolation)

        self.spec_time_line.set_data(payload.time_x, payload.time_y)
        self.spec_side_line.set_data(payload.freq_x, payload.freq_y)
        self.spec_side_line.set_color(payload.freq_color)
        self.spec_time_info_text.set_text(payload.time_info)
        self.spec_info_text.set_text(payload.info_text)
        self.ax_spec_time.set_ylabel(payload.time_ylabel, fontsize=AXIS_LABEL_FONTSIZE)

        if payload.toa is None:
            self.spec_toa_line.set_visible(False)
            self.spec_legend.set_visible(False)
        else:
            self.spec_toa_line.set_visible(True)
            self.spec_toa_line.set_xdata([payload.toa, payload.toa])
            self.spec_legend.set_visible(True)
            if self.spec_legend.texts:
                self.spec_legend.texts[0].set_text(f"TOA: {payload.toa:.3f}s")

        self.ax_spec_main.set_ylabel(payload.ylabel, fontsize=AXIS_LABEL_FONTSIZE)
        self.ax_spec_main.set_xlabel(payload.xlabel, fontsize=AXIS_LABEL_FONTSIZE, labelpad=8)
        self.ax_spec_main.set_xlim(*payload.xlim)
        self.ax_spec_main.set_ylim(*payload.ylim)
        self.ax_spec_time.set_xlim(*payload.xlim)
        self.ax_spec_side.set_ylim(*payload.ylim)
        _autoscale_axis(self.ax_spec_time, scale_x=False, scale_y=True)

        self.ax_spec_side.set_title(payload.freq_title, fontsize=AXIS_LABEL_FONTSIZE, pad=6)
        self.ax_spec_side.set_xlim(*payload.freq_xlim)
        self.ax_spec_side.set_xticks([payload.freq_xlim[0], payload.freq_xlim[1]])
        self.ax_spec_side.set_xticklabels(["0", "1"], fontsize=AXIS_TICK_FONTSIZE)

    def _clear_spectrum(self) -> None:
        blank = np.zeros((1, 1), dtype=np.float32)
        self.spec_image.set_data(blank)
        self.spec_image.set_extent([0, 1, 0, 1])
        self.spec_image.set_clim(0, 1)
        self.spec_time_line.set_data([], [])
        self.spec_side_line.set_data([], [])
        self.spec_toa_line.set_visible(False)
        self.spec_legend.set_visible(False)
        self.spec_time_info_text.set_text("")
        self.spec_info_text.set_text("")
        self.ax_spec_main.set_xlim(0, 1)
        self.ax_spec_main.set_ylim(0, 1)
        self.ax_spec_time.set_xlim(0, 1)
        self.ax_spec_side.set_ylim(0, 1)


def _autoscale_axis(ax, scale_x: bool, scale_y: bool) -> None:
    ax.relim()
    ax.autoscale_view(scalex=scale_x, scaley=scale_y)


def _single_dm_axes(fig) -> dict[str, object]:
    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[3, 1],
        height_ratios=[1, 3],
        wspace=0.07,
        hspace=0.04,
    )
    ax_time = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0], sharex=ax_time)
    ax_side = fig.add_subplot(gs[1, 1], sharey=ax_main)
    return {"time": ax_time, "main": ax_main, "side": ax_side}


def _single_spec_axes(fig) -> dict[str, object]:
    gs = GridSpec(
        2,
        2,
        figure=fig,
        width_ratios=[3, 1],
        height_ratios=[1, 3],
        wspace=0.07,
        hspace=0.04,
    )
    ax_time = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0], sharex=ax_time)
    ax_side = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_info = fig.add_subplot(gs[0, 1])
    return {"time": ax_time, "main": ax_main, "side": ax_side, "info": ax_info}


def _combined_axes(fig) -> tuple[dict[str, object], dict[str, object]]:
    gs = GridSpec(
        2,
        5,
        figure=fig,
        width_ratios=[3, 1, 0.29, 3, 1],
        height_ratios=[1, 3],
        wspace=0.06,
        hspace=0.04,
    )
    ax_dm_time = fig.add_subplot(gs[0, 0])
    ax_dm_main = fig.add_subplot(gs[1, 0], sharex=ax_dm_time)
    ax_dm_side = fig.add_subplot(gs[1, 1], sharey=ax_dm_main)

    ax_spec_time = fig.add_subplot(gs[0, 3])
    ax_spec_main = fig.add_subplot(gs[1, 3], sharex=ax_spec_time)
    ax_spec_side = fig.add_subplot(gs[1, 4], sharey=ax_spec_main)
    ax_spec_info = fig.add_subplot(gs[0, 4])

    return (
        {"time": ax_dm_time, "main": ax_dm_main, "side": ax_dm_side},
        {
            "time": ax_spec_time,
            "main": ax_spec_main,
            "side": ax_spec_side,
            "info": ax_spec_info,
        },
    )
