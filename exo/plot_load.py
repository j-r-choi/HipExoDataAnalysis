"""Load-cell raw ADC counts for both legs."""
import matplotlib.pyplot as plt


def plot_load(df, time_range=None):
    t = df["time"]
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(t, df["left_load"], label="Left")
    ax.plot(t, df["right_load"], label="Right")
    ax.set_ylabel("Load (raw ADC)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc="best")

    if time_range:
        ax.set_xlim(time_range)

    fig.suptitle("Load cell (raw ADC counts)")
    fig.tight_layout()
    return fig, ax
