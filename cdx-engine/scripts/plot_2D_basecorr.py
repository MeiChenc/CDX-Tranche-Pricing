import matplotlib.pyplot as plt

# x: detachment rate, y: base correlation (rho)
basecorr_by_tenor = {
    1.0:  {0.03: 0.017807, 0.07: 0.998999, 0.10: 0.998999, 0.15: 0.998999},
    2.0:  {0.03: 0.00654819631483541, 0.07: 0.703090722059275, 0.10: 0.7248070870215905, 0.15: 0.7836233400696122, },
    3.0:  {0.03: 0.003512664571331765, 0.07: 0.007321840795875427, 0.10: 0.02180166745509742, 0.15: 0.02657520188875312,},
    5.0:  {0.03: 0.0001596912618323064, 0.07: 0.004883747247324193, 0.10: 0.008547697046573228, 0.15: 0.010442534988935047, },
    7.0:  {0.03: 0.00010053538642225109, 0.07: 0.0007266960107467071, 0.10: 0.0015878989474963285, 0.15: 0.00822749208213126,},
    10.0: {0.03: 0.00010053538642225109, 0.07: 0.00010053538642225109, 0.10: 0.06222877921866875, 0.15: 0.06469981296890935, },
}

plt.figure(figsize=(9, 5))

for tenor in sorted(basecorr_by_tenor.keys()):
    pts = basecorr_by_tenor[tenor]
    xs = sorted(pts.keys())
    ys = [pts[x] for x in xs]
    plt.plot(xs, ys, marker="o", label=f"{tenor:.1f}Y")

plt.title("Base Correlation vs Detachment (6 tenors)")
plt.xlabel("Detachment rate")
plt.ylabel("Base correlation (rho)")
plt.ylim(-0.02, 1.02)     # 你有 rho 接近 1
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
