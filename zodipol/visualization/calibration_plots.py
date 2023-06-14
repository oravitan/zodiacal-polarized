import numpy as np
import matplotlib.pyplot as plt


def plot_deviation_comp(parser, polarizance_real, polarizance_est_reshape, saveto=None):
    """
    Plot the deviation of the estimated polarizance from the real polarizance.
    :param parser: parser object
    :param polarizance_real: real polarizance of the observation (GT)
    :param polarizance_est_reshape: estimated polarizance of the observation
    :param saveto: path to save the plot
    """
    pol_mean_deviation = polarizance_real
    pol_est_mean_deviation = polarizance_est_reshape
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    p1, p2 = pol_mean_deviation.squeeze(), pol_est_mean_deviation

    c1 = ax1.imshow(p1.reshape(parser["resolution"]))
    c2 = ax2.imshow(p2.reshape(parser["resolution"]))
    cbar1 = fig.colorbar(c1, ax=ax1); cbar1.ax.tick_params(labelsize=14)
    cbar2 = fig.colorbar(c2, ax=ax2); cbar2.ax.tick_params(labelsize=14)
    ax1.set_title('True $P$', fontsize=18); ax1.set_axis_off()
    ax2.set_title('$\hat{P}$', fontsize=18); ax2.set_axis_off()
    fig.tight_layout()
    if saveto is not None:
        plt.savefig(saveto, format='pdf', bbox_inches='tight', transparent="True", pad_inches=0)
    plt.show()


def plot_mueller(mueller, parser, cbar=False, saveto=None, vmin=None, vmax=None):
    """
    Plot a Mueller matrix.
    :param mueller: Mueller matrix to plot, shape (..., :3, :3) or (..., :4, :4)
    :param parser: parser object (for resolution)
    :param cbar: whether to plot a colorbar
    :param saveto: path to save the plot
    :param vmin: min value for the colorbar (default: min value of the matrix)
    :param vmax: max value for the colorbar (default: max value of the matrix)
    """
    if vmin is None:
        vmin = np.nanmin(mueller)
    if vmax is None:
        vmax = np.nanmax(mueller)
    mueller = mueller[..., :3, :3]
    fig, ax = plt.subplots(3,3, figsize=(6,6), sharex='col', sharey='row', subplot_kw={'xticks': [], 'yticks': []})
    for i in range(3):
        for j in range(3):
            c = ax[i,j].imshow(mueller[..., i, j].reshape(parser["resolution"]), vmin=vmin, vmax=vmax)
            # ax[i,j].get_xaxis().set_visible(False)
            # ax[i,j].get_yaxis().set_visible(False)
    ax[0,0].set_ylabel(0, fontsize=16)
    ax[1,0].set_ylabel(1, fontsize=16)
    ax[2,0].set_ylabel(2, fontsize=16)
    ax[2,0].set_xlabel(0, fontsize=16)
    ax[2,1].set_xlabel(1, fontsize=16)
    ax[2,2].set_xlabel(2, fontsize=16)
    # fig.colorbar(c, ax=ax.ravel().tolist())
    if cbar:
        cb = fig.colorbar(c, ax=ax.ravel().tolist())
        cb.ax.tick_params(labelsize=14)
    else:
        plt.tight_layout(w_pad=-15.0, h_pad=1.0)
    if saveto is not None:
        plt.savefig(saveto, format='pdf', bbox_inches='tight', transparent="True", pad_inches=0)
    plt.show()


def plot_cost_itr(cost_itr, p_cost, mueller_cost, saveto=None):
    """
    Plot the cost function values for each iteration.
    :param cost_itr: cost function values for each iteration
    :param p_cost: polarizance cost function values for each iteration
    :param mueller_cost: Mueller matrix cost function values for each iteration
    :param saveto: path to save the plot
    """
    fig, ax = plt.subplots(3, 1, figsize=(6, 5), sharex=True)
    ax[0].plot(cost_itr, lw=3)
    ax[0].set_ylabel('Intensity RMSE\n($electrons^2$)', fontsize=16)
    ax[0].tick_params(labelsize=16)
    ax[0].grid()
    ax[0].set_ylim(0, None)

    ax[1].plot(p_cost, lw=3)
    ax[1].set_ylabel('$\hat{P}$ RMSE', fontsize=16)
    ax[1].tick_params(labelsize=16)
    ax[1].grid()
    ax[1].set_ylim(0, None)

    ax[2].plot(mueller_cost, lw=3)
    ax[2].set_ylabel('$\hat{B}$ RMSE', fontsize=16)
    ax[2].tick_params(labelsize=16)
    ax[2].grid()
    ax[2].set_xlabel('Iteration number', fontsize=16)
    ax[2].set_ylim(0, None)

    fig.tight_layout()
    if saveto is not None:
        plt.savefig(saveto, format='pdf', bbox_inches='tight', transparent="True", pad_inches=0)
    plt.show()


def plot_res_comp_plot(x_value, p_mse, biref_mse, saveto=None, xlabel=None, ylim1=None, ylim2=None, type='RMSE',
                       p_mse_err=None, biref_mse_err=None):
    """
    Plot the residual comparison plot.
    :param x_value: x-axis values
    :param p_mse: polarizance RMSE values
    :param biref_mse: Mueller matrix RMSE values
    :param saveto: path to save the plot
    :param xlabel: x-axis label
    :param ylim1: y-axis limits for polarizance RMSE
    :param ylim2: y-axis limits for Mueller matrix RMSE
    :param type: 'RMSE' or another string as function type
    """
    fig, ax = plt.subplots(1, 1, figsize=(6, 2))
    ax.errorbar(x_value, p_mse, yerr=p_mse_err, lw=2, c='b', capsize=3)
    ax2 = ax.twinx()
    ax2.errorbar(x_value, biref_mse, yerr=biref_mse_err, lw=2, c='r', capsize=3)  # ax2.semilogy(x_value, biref_mse, lw=2, c='r')
    ax.grid()
    ax.set_xlabel(xlabel, fontsize=16)
    ax.tick_params(axis='x', labelsize=16)
    ax.tick_params(axis='y', colors='b', labelsize=16)
    ax2.tick_params(axis='y', colors='r', labelsize=16)
    if type == 'RMSE':
        ax.set_ylabel("$\\left\\Vert\\hat{P}-P\\right\\Vert_{2}^{1}$", fontsize=16, c='b', rotation='horizontal', labelpad=35)
        ax2.set_ylabel("$\\left\\Vert\\hat{\\bf B}-{\\bf B}\\right\\Vert_{F}^{1}$", fontsize=16, c='r', rotation='horizontal', labelpad=35)
    else:
        ax.set_ylabel(type + "($\\hat{P}-P$)", fontsize=16, c='b', rotation='horizontal', labelpad=35)
        ax2.set_ylabel(type + "($\\hat{\\bf B}-{\\bf B}$)", fontsize=16, c='r', rotation='horizontal', labelpad=35)
    fig.tight_layout()
    if ylim1 is not None:
        ax.set_ylim(*ylim1)
    if ylim2 is not None:
        ax2.set_ylim(*ylim2)
    if saveto is not None:
        plt.savefig(saveto, format='pdf', bbox_inches='tight', transparent="True", pad_inches=0)
    plt.show()
