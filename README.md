G–VG + Switching Correlation Model for Multi-Tenor CDX Tranche Calibration

This repository implements a Generalized Variance-Gamma (G–VG) mixture copula with switching correlation to calibrate CDX index tranche spreads across all maturities and construct the Gaussian-equivalent base correlation surface.

The model is designed for production-grade quantitative credit modeling, capturing systemic heavy tails, default clustering, and state-dependent correlation dynamics.

1. Objective

The goal is to reproduce market tranche spreads by calibrating a flexible dependence structure that accounts for:

Heavy-tailed systemic shocks

Correlation jumps in stress regimes

Maturity-specific correlation dynamics

Consistency with market-implied survival curves

The outputs include:

Per-maturity calibrated dependence parameters

Model tranche PVs and implied spreads

Gaussian-equivalent base correlation surface

2. Model Architecture

The dependence structure is driven by two latent layers:

2.1 Systemic Factor – G–VG Mixture

The systemic factor is:

𝑌
∼
𝑝
(
𝑇
)
 
𝑁
(
0
,
1
)
+
(
1
−
𝑝
(
𝑇
)
)
 
𝑉
𝐺
(
𝜆
,
𝛼
,
𝛽
,
𝜇
)
Y∼p(T)N(0,1)+(1−p(T))VG(λ,α,β,μ)

𝑝
(
𝑇
)
p(T): probability of normal regime (calibrated per maturity)

𝑉
𝐺
(
⋅
)
VG(⋅): Variance-Gamma distribution controlling heavy-tail stress

VG parameters 
(
𝜆
,
𝛼
,
𝛽
,
𝜇
)
(λ,α,β,μ): global fixed parameters

VG Sampling (Gamma-mixture representation)
𝑌
=
𝜇
+
𝛽
𝐺
+
𝛼
𝐺
𝑍
Y=μ+βG+α
G
	​

Z

with

𝑍
∼
𝑁
(
0
,
1
)
Z∼N(0,1)

𝐺
∼
G
a
m
m
a
(
𝑐
,
1
/
𝑐
)
G∼Gamma(c,1/c)

This method avoids scipy.variance_gamma and is numerically stable.

2.2 Switching Correlation Layer

The correlation regime is driven by another latent variable:

𝑍
𝜌
∼
𝐵
𝑒
𝑟
(
𝑝
′
(
𝑇
)
)
Z
ρ
	​

∼Ber(p
′
(T))
𝜌
(
𝑇
)
=
{
𝜌
𝐻
(
𝑇
)
,
	
𝑍
𝜌
=
1


𝜂
(
𝑇
)
,
	
𝑍
𝜌
=
0
ρ(T)={
ρ
H
	​

(T),
η(T),
	​

Z
ρ
	​

=1
Z
ρ
	​

=0
	​


𝑝
′
(
𝑇
)
p
′
(T): probability of high-correlation stress

𝜌
𝐻
(
𝑇
)
ρ
H
	​

(T): correlation level under systemic stress

𝜂
(
𝑇
)
η(T): benign low-correlation level

This produces realistic correlation jumps and term-structure behavior.

3. Default Model

Each name 
𝑗
j has latent variable:

𝑋
𝑗
=
𝜌
 
𝑌
+
1
−
𝜌
 
𝜀
𝑗
,
𝜀
𝑗
∼
𝑁
(
0
,
1
)
X
j
	​

=
ρ
	​

Y+
1−ρ
	​

ε
j
	​

,ε
j
	​

∼N(0,1)

Default occurs if 
𝑋
𝑗
≤
𝑥
crit
(
𝑇
)
X
j
	​

≤x
crit
	​

(T).

The threshold satisfies:

Pr
⁡
(
𝑋
𝑗
≤
𝑥
crit
)
=
1
−
𝑒
−
𝜆
(
𝑇
)
𝑇
Pr(X
j
	​

≤x
crit
	​

)=1−e
−λ(T)T

where 
𝜆
(
𝑇
)
λ(T) is the bootstrapped index hazard rate.

Threshold is solved via integral root search:

𝑝
default
(
𝑇
)
=
𝐸
𝑌
 ⁣
[
Φ
 ⁣
(
𝑥
crit
−
𝜌
𝑌
1
−
𝜌
)
]
p
default
	​

(T)=E
Y
	​

[Φ(
1−ρ
	​

x
crit
	​

−
ρ
	​

Y
	​

)]
4. Survival Curve Construction

For each maturity 
𝑇
T:

Bootstrap a flat hazard rate 
𝜆
(
𝑇
)
λ(T)

Match index CDS PV using protection/premium leg equality

P
V
p
r
o
t
(
𝜆
)
=
P
V
p
r
e
m
(
𝜆
;
𝑠
𝑇
)
PV
prot
	​

(λ)=PV
prem
	​

(λ;s
T
	​

)

This yields:

𝑝
default
(
𝑇
)
=
1
−
𝑒
−
𝜆
(
𝑇
)
𝑇
p
default
	​

(T)=1−e
−λ(T)T

These probabilities are used to solve 
𝑥
crit
x
crit
	​

.

5. Tranche Pricing

For each tranche 
[
𝐾
1
,
𝐾
2
]
[K
1
	​

,K
2
	​

] and maturity 
𝑇
T:

Draw latent regime:

𝑌
∼
Y∼ G–VG mixture

𝜌
=
𝜌
𝐻
(
𝑇
)
ρ=ρ
H
	​

(T) or 
𝜂
(
𝑇
)
η(T)

Compute conditional default probability:

𝑝
(
𝑦
)
=
Φ
 ⁣
(
𝑥
crit
−
𝜌
 
𝑦
1
−
𝜌
)
p(y)=Φ(
1−ρ
	​

x
crit
	​

−
ρ
	​

y
	​

)

Simulate defaults across the homogeneous pool

Compute:

Expected Tranche Loss (EL)

Risky PV01 (RP)

Model running spread:

𝑠
𝑚
𝑜
𝑑
𝑒
𝑙
=
𝐸
𝐿
𝑅
𝑃
×
10,000
 bps
s
model
=
RP
EL
	​

×10,000 bps

Monte Carlo paths:

300k for pricing

Vectorized for performance

6. Calibration

For each maturity 
𝑇
T, calibrate:

𝑝
(
𝑇
)
,
𝑝
′
(
𝑇
)
,
𝜌
𝐻
(
𝑇
)
,
𝜂
(
𝑇
)
p(T),p
′
(T),ρ
H
	​

(T),η(T)
	​


Objective:

min
⁡
𝜃
(
𝑇
)
∑
𝑘
𝑤
𝑘
(
𝑠
𝑘
𝑚
𝑜
𝑑
𝑒
𝑙
(
𝑇
)
−
𝑠
𝑘
𝑚
𝑎
𝑟
𝑘
𝑒
𝑡
(
𝑇
)
)
2
θ(T)
min
	​

k
∑
	​

w
k
	​

(s
k
model
	​

(T)−s
k
market
	​

(T))
2

where 
𝑘
k indexes tranches
and 
𝑤
𝑘
w
k
	​

 upweights equity tranche.

VG parameters 
(
𝜆
,
𝛼
,
𝛽
,
𝜇
)
(λ,α,β,μ) remain global fixed.

7. Base Correlation Surface

For each maturity 
𝑇
T and base detachment 
𝐾
K:

Solve for Gaussian copula correlation 
𝜌
𝐺
(
𝑇
,
𝐾
)
ρ
G
	​

(T,K):

𝑃
𝑉
Gauss
(
𝑇
,
𝐾
;
𝜌
𝐺
)
=
𝑃
𝑉
GVG
(
𝑇
,
𝐾
;
𝜃
^
𝑇
)
PV
Gauss
	​

(T,K;ρ
G
	​

)=PV
GVG
	​

(T,K;
θ
^
T
	​

)

This produces a full base correlation surface
compatible with standard trading-desk risk systems.

8. Parameter Summary
Global Fixed Parameters (Not Calibrated)
Parameter	Meaning

𝜆
λ	VG shape

𝛼
α	VG scale (vol component)

𝛽
β	VG skew

𝜇
μ	VG location

These control systemic heavy-tail behavior.

Per-Maturity Calibrated Parameters
Parameter	Meaning

𝑝
(
𝑇
)
p(T)	Normal vs heavy-tail mixture weight

𝑝
′
(
𝑇
)
p
′
(T)	High-correlation regime probability

𝜌
𝐻
(
𝑇
)
ρ
H
	​

(T)	Stress correlation level

𝜂
(
𝑇
)
η(T)	Low-correlation level

These shape the dependence structure required to fit market tranche spreads.

9. Advantages

Captures heavy tails and systemic clustering

Supports correlation jumps during crises

Produces realistic maturity term structure

More flexible than Gaussian & t-copula

Stable calibration and interpretation

10. Repository Structure
/code
    gvg_model.py                # Core G–VG mixture copula implementation
    tranche_pricer.py           # Monte Carlo tranche pricer
    calibration.py              # Per-maturity calibration routines
/data
    cdx_market_data_multi_tenor.json
/output
    base_correlation_surface.csv
README.md

11. References

Li (2000), Gaussian Copula model

Madan & Seneta (1990), Variance-Gamma processes

Duffie & Singleton (2003), Credit Risk Modeling

Market practice from CDO/tranche desks (JPM, Citi, BAML)
