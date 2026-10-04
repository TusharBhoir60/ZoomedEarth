import numpy as np, rasterio
d="data/raw/sentinel2/S2B_43RFM_20230203_0_L2A"
print(open(f"{d}/metadata.json").read())
for b in ["B02","B04","B08"]:
    with rasterio.open(f"{d}/{b}.tif") as s:
        a=s.read(1,out_shape=(s.height//10,s.width//10)).astype("float64")
    v=a[a>0]; print(b,s.dtypes[0],"DN==0 %.1f%%"%((a==0).mean()*100),"DN p0.1/1/5/50:",np.percentile(v,[0.1,1,5,50]).round(0))

from pystac_client import Client
it=next(Client.open("https://earth-search.aws.element84.com/v1").search(collections=["sentinel-2-l2a"],ids=["S2B_43RFM_20230203_0_L2A"]).items())
p=it.properties
print({k:p.get(k) for k in ["s2:processing_baseline","earthsearch:boa_offset_applied","updated"]})
print(it.assets["red"].extra_fields.get("raster:bands")); print(it.assets["nir"].extra_fields.get("raster:bands"))
