import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
gs = gridspec.GridSpec(8, 8)
############################################################################################
m1 = 28
m2 = 2
net_profit = str(15.324)

x = np.arange(1, 10)
y = np.arange(2, 11)

ax1 = plt.subplot(gs[0:2, 0:3],facecolor = 'k')
ax2 = plt.subplot(gs[3:5, 0:3],facecolor = 'k')
ax3 = plt.subplot(gs[1:4, 4:],facecolor = 'k')
ax4 = plt.subplot(gs[5:8, 4:],facecolor = 'k')


ax1.plot(x, y)
ax2.plot(x, y)
ax3.plot(x, y)
ax4.plot(x, y)
#############################################################################################
ax1.set_title("Algorithmic Trading Bot\n\n\nModel 1")
ax1.set_ylabel('Price [$]')
ax1.set_xlabel('Time [ticks]')
ax1.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    left= False,
    right=False,
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off

ax2.set_title("Model 2")
ax2.set_ylabel('Price [$]')
ax2.set_xlabel('Time [ticks]')
ax2.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    left= False,
    right=False,
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off

ax3.set_title("Final Buy/Sell Decision ")
ax3.set_ylabel('Price [$]')
ax3.set_xlabel('Time [ticks]')
ax3.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    left= False,
    right=False,
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off

ax4.set_title("Net Profit")
ax4.set_ylabel('Price [$]')
ax4.set_xlabel('Time [ticks]')
ax4.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    left= False,
    right=False,
    top=False,         # ticks along the top edge are off
    labelbottom=True) # labels along the bottom edge are off

ax1.text(0.99, 0.01, m1,
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax1.transAxes,
        color='green', fontsize=10)

ax2.text(0.99, 0.01, m2,
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax2.transAxes,
        color='green', fontsize=10)


label_profit = 'Profit: '
ax4.text(0.99, 0.01, label_profit + net_profit,
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax4.transAxes,
        color='green', fontsize=10)

plt.show()