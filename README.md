# **XeDist**
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/rob-tay/XeDist/master)

Calculate the 136Xe 0nBB background distribution in a LXe TPC. Used in my thesis to predict the external background in a generation-3 TPC without detailed Monte-Carlo. 

Uses a mathematical model of gamma-ray attenuation, and the integrated rate from LZ (https://arxiv.org/abs/1912.04248) 

### Usage

```python
import xedist

tpc = xedist.TPC(radius=72.8, height=145.6)

pmt_source = xedist.DiskGammaSource(tpc.radius, 0, energy=2447, n=10)

dist = xedist.spatial_distribution(tpc, pmt_source)
cumulative_rate = xedist.get_cumulative_rate(dist, tpc)
```
### Notebooks
Predictions for a G3 experiment are [here](g3.ipynb)

See [validation](validation.ipynb) for comparisons with LZ and DARWIN (https://arxiv.org/pdf/2003.13407.pdf). 

