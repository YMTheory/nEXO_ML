
def setaxis(ax, xlabel="x", ylabel="y", title="tl", xlabelcolor="black", ylabelcolor="black", xlabelsize=13, ylabelsize=13, xticksize=12, yticksize=12, titlesize=13, lg=False, lgloc="best", lgsize=12, ncol=1):
    ax.set_xlabel(xlabel, fontsize=xlabelsize, color=xlabelcolor)
    ax.set_ylabel(ylabel, fontsize=ylabelsize, color=ylabelcolor)
    ax.tick_params(axis="x", labelsize=xticksize)
    ax.tick_params(axis="y", labelsize=yticksize)
    ax.set_title(title, fontsize=titlesize)
    if lg:
        ax.legend(prop={"size":lgsize}, loc=lgloc, ncol=ncol, frameon=True)


def setcbar(cb, label="z", labelsize=13, ticksize=12):
    cb.set_label(label, fontsize=labelsize)
    cb.ax.tick_params(labelsize=ticksize)
