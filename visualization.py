"""
시각화 모듈

논문의 주요 결과를 시각화한다:
1. 동적 TCI (Total Connectedness Index) 시계열
2. 순방향 연결성 (Net Directional Connectedness) 시계열
3. 평균 GFEVD 히트맵
4. 동적 헤지비율 시계열
5. 포트폴리오 가중치 변화
6. 포트폴리오 누적 수익률
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import os

OUTPUT_DIR = "figures"


def setup_plot_style():
    """논문 스타일 플롯 설정."""
    plt.rcParams.update({
        "figure.figsize": (12, 6),
        "figure.dpi": 150,
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.2,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_tci(tci_series, save=True):
    """동적 Total Connectedness Index 시계열을 그린다."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(14, 5))

    ax.plot(tci_series.index, tci_series.values, color="navy", linewidth=1.0)
    ax.fill_between(tci_series.index, tci_series.values, alpha=0.15, color="navy")
    ax.set_title("Dynamic Total Connectedness Index (TCI)")
    ax.set_ylabel("TCI (%)")
    ax.set_xlabel("Date")

    # 평균선
    avg = tci_series.mean()
    ax.axhline(y=avg, color="red", linestyle="--", linewidth=0.8, label=f"Mean: {avg:.1f}%")
    ax.legend()

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, "tci_dynamic.png"), bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR}/tci_dynamic.png")
    plt.close()


def plot_net_connectedness(net_df, save=True):
    """순방향 연결성 시계열을 그린다."""
    setup_plot_style()
    fig, axes = plt.subplots(len(net_df.columns), 1, figsize=(14, 3 * len(net_df.columns)),
                              sharex=True)

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, col in enumerate(net_df.columns):
        ax = axes[idx] if len(net_df.columns) > 1 else axes
        values = net_df[col].values
        dates = net_df.index

        ax.fill_between(dates, values, where=(values >= 0), alpha=0.4,
                        color=colors[idx % len(colors)], label="Net Transmitter")
        ax.fill_between(dates, values, where=(values < 0), alpha=0.4,
                        color="gray", label="Net Receiver")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.set_ylabel(f"{col}\nNET (%)")
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)

    fig.suptitle("Net Directional Connectedness", fontsize=14, y=1.01)
    plt.tight_layout()

    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, "net_connectedness.png"), bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR}/net_connectedness.png")
    plt.close()


def plot_gfevd_heatmap(gfevd_table, save=True):
    """평균 GFEVD 테이블 히트맵을 그린다."""
    setup_plot_style()

    # FROM, TO, NET 행/열 제외한 N×N 부분만 히트맵
    cols = [c for c in gfevd_table.columns if c not in ["FROM"]]
    rows = [r for r in gfevd_table.index if r not in ["TO", "NET"]]
    data = gfevd_table.loc[rows, cols].astype(float)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(data, annot=True, fmt=".1f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, cbar_kws={"label": "%"})
    ax.set_title("Average Connectedness Table (GFEVD, %)")
    ax.set_ylabel("FROM ←")
    ax.set_xlabel("TO →")
    plt.tight_layout()

    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, "gfevd_heatmap.png"), bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR}/gfevd_heatmap.png")
    plt.close()


def plot_from_to_bar(gfevd_table, save=True):
    """FROM/TO/NET 막대 그래프를 그린다."""
    setup_plot_style()
    rows = [r for r in gfevd_table.index if r not in ["TO", "NET"]]

    from_vals = gfevd_table.loc[rows, "FROM"].values.astype(float)
    to_vals = gfevd_table.loc["TO", rows].values.astype(float)
    net_vals = to_vals - from_vals

    x = np.arange(len(rows))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, from_vals, width, label="FROM others", color="#2196F3")
    ax.bar(x, to_vals, width, label="TO others", color="#FF9800")
    ax.bar(x + width, net_vals, width, label="NET", color="#4CAF50")
    ax.axhline(y=0, color="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(rows)
    ax.set_ylabel("%")
    ax.set_title("Directional Connectedness: FROM / TO / NET")
    ax.legend()
    plt.tight_layout()

    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, "from_to_net.png"), bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR}/from_to_net.png")
    plt.close()


def plot_hedge_ratios(hedge_ratios, index, pairs=None, save=True):
    """동적 헤지비율 시계열을 그린다."""
    setup_plot_style()

    if pairs is None:
        pairs = list(hedge_ratios.keys())[:6]

    n_pairs = len(pairs)
    fig, axes = plt.subplots(n_pairs, 1, figsize=(14, 3 * n_pairs), sharex=True)
    if n_pairs == 1:
        axes = [axes]

    for idx, pair in enumerate(pairs):
        ax = axes[idx]
        values = hedge_ratios[pair]
        n = min(len(values), len(index))
        values = values[-n:]
        dates = index[-n:]
        ax.plot(dates, values, linewidth=0.8, color="navy")
        avg = np.mean(values)
        ax.axhline(y=avg, color="red", linestyle="--", linewidth=0.6,
                    label=f"Mean: {avg:.4f}")
        ax.set_ylabel(pair)
        ax.legend(loc="upper right", fontsize=8)

    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)

    fig.suptitle("Dynamic Hedge Ratios (Kroner & Sultan, 1993)", fontsize=14, y=1.01)
    plt.tight_layout()

    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, "hedge_ratios.png"), bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR}/hedge_ratios.png")
    plt.close()


def plot_portfolio_weights(w_dict, index, columns, save=True):
    """동적 포트폴리오 가중치 변화를 스택 영역 차트로 그린다."""
    setup_plot_style()

    portfolio_names = ["MVP", "MCP", "MCoP"]
    weight_keys = ["w_mvp", "w_mcp", "w_mcop"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, axes = plt.subplots(len(portfolio_names), 1, figsize=(14, 4 * len(portfolio_names)),
                              sharex=True)

    for idx, (name, key) in enumerate(zip(portfolio_names, weight_keys)):
        ax = axes[idx]
        w = w_dict[key]
        n = min(w.shape[0], len(index))
        w = w[-n:]
        dates = index[-n:]

        ax.stackplot(dates, w.T, labels=columns, colors=colors[:len(columns)], alpha=0.8)
        ax.set_ylabel("Weight")
        ax.set_title(f"{name} - Dynamic Portfolio Weights")
        ax.set_ylim(0, 1)
        ax.legend(loc="upper right", fontsize=8, ncol=len(columns))

    axes[-1].xaxis.set_major_locator(mdates.YearLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, "portfolio_weights.png"), bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR}/portfolio_weights.png")
    plt.close()


def plot_cumulative_returns(port_results, index, save=True):
    """포트폴리오 누적 수익률을 그린다."""
    setup_plot_style()

    fig, ax = plt.subplots(figsize=(14, 6))

    names = ["Equal Weight (1/N)", "MVP", "MCP", "MCoP"]
    keys = ["r_equal", "r_mvp", "r_mcp", "r_mcop"]
    colors = ["gray", "#1f77b4", "#ff7f0e", "#2ca02c"]
    styles = ["--", "-", "-", "-"]

    for name, key, color, style in zip(names, keys, colors, styles):
        r = port_results[key]
        n = min(len(r), len(index))
        r = r[-n:]
        dates = index[-n:]
        cum_r = np.cumsum(r)
        ax.plot(dates, cum_r, color=color, linestyle=style, label=name, linewidth=1.2)

    ax.set_title("Cumulative Portfolio Returns")
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_xlabel("Date")
    ax.legend()
    ax.axhline(y=0, color="black", linewidth=0.3)

    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    plt.xticks(rotation=45)
    plt.tight_layout()

    if save:
        fig.savefig(os.path.join(OUTPUT_DIR, "cumulative_returns.png"), bbox_inches="tight")
        print(f"Saved: {OUTPUT_DIR}/cumulative_returns.png")
    plt.close()


def generate_all_figures(conn_results, hr_results, port_results, returns):
    """모든 그림을 생성한다."""
    setup_plot_style()

    index = conn_results["TCI"].index
    columns = conn_results["columns"]

    print("\n=== Generating Figures ===")
    plot_tci(conn_results["TCI"])
    plot_net_connectedness(conn_results["NET"])
    plot_gfevd_heatmap(conn_results["GFEVD_table"])
    plot_from_to_bar(conn_results["GFEVD_table"])

    if hr_results:
        plot_hedge_ratios(hr_results["hedge_ratios"], index)

    if port_results:
        plot_portfolio_weights(port_results, index, columns)
        plot_cumulative_returns(port_results, index)

    print(f"\nAll figures saved to '{OUTPUT_DIR}/' directory.")
