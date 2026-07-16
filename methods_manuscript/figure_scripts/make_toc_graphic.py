from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

out = Path(__file__).resolve().parents[1]
fig, ax = plt.subplots(figsize=(3.25, 1.72))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5.2)
ax.axis('off')

navy = '#1F4E79'; teal = '#2A7F62'; orange = '#B45F06'
light1 = '#EAF2F8'; light2 = '#E8F5EF'; light3 = '#FFF2DF'; dark = '#222222'

def panel(x, face, edge, title):
    w, y, h = 2.65, 0.70, 3.95
    p = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.06,rounding_size=0.14',
                       linewidth=1.05, facecolor=face, edgecolor=edge)
    ax.add_patch(p)
    ax.text(x+w/2, y+h-0.33, title, ha='center', va='top', fontsize=6.1,
            fontweight='bold', color=edge, linespacing=1.0)
    return w, y, h

w,y,h = panel(0.10, light1, navy, 'MPS roots\n& response')
for i in range(5):
    cx = 0.64 + i*0.39
    ax.add_patch(Circle((cx, 2.65), 0.095, facecolor=navy, edgecolor='white', linewidth=0.45))
    if i < 4:
        ax.plot([cx+0.095, cx+0.295], [2.65, 2.65], color=navy, lw=1.0)
ax.text(1.425, 1.35, 'no dense active-space\nresponse vector', ha='center', va='center', fontsize=4.9, color=dark)

w,y,h = panel(3.68, light2, teal, 'Independent\nfinite differences')
ax.text(5.005, 2.90, r'$\partial E/\partial R$', ha='center', va='center', fontsize=8.0, color=teal)
ax.text(5.005, 2.35, r'$\langle\Psi(R_0)|\Psi(R_\pm)\rangle$', ha='center', va='center', fontsize=5.7, color=teal)
ax.text(5.005, 1.35, 'same DMRG surface\n+ direct MPS overlap', ha='center', va='center', fontsize=4.9, color=dark)

w,y,h = panel(7.25, light3, orange, 'Beyond dense FCI\n& repeated calls')
ax.text(8.575, 2.92, r'$3.41\times10^{10}$', ha='center', va='center', fontsize=7.3,
        fontweight='bold', color=orange)
ax.text(8.575, 2.50, r'$M_S=0$ determinants', ha='center', va='center', fontsize=4.6, color=orange)
ax.text(8.575, 1.42, 'CAS(20,20) check\n715 SHARC frames', ha='center', va='center', fontsize=4.9, color=dark)

for x1, x2 in [(2.82, 3.60), (6.40, 7.17)]:
    ax.add_patch(FancyArrowPatch((x1, 2.62), (x2, 2.62), arrowstyle='-|>',
                                 mutation_scale=10, linewidth=1.15, color='#444444'))

ax.text(5.0, 0.23, 'Exact controls where available; continuity and residual diagnostics reported explicitly',
        ha='center', va='center', fontsize=4.1, color='#333333')

fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.01)
fig.savefig(out / 'toc_graphic.pdf', bbox_inches='tight', pad_inches=0.015)
fig.savefig(out / 'toc_graphic.png', dpi=600, bbox_inches='tight', pad_inches=0.015)
plt.close(fig)
