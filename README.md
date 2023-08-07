```
import alma_var
av = alma_var.AlmaVar(ms='path/to/data.ms',
                      outdir='path/to/output')
av.process()
```

Each input ms file results in a folder with the time series for each scan, plus a handful of other output, e.g. sanity checks on data weights.

Scans with significant variation are linked in a folder named for the target observed. 