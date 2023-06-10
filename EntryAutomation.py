from tqdm import tqdm
import logging
from pathlib import Path
from osgeo import gdal, ogr

logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)

if __name__ == '__main__':
    in_dir = r'D:\Capstone Data\F03'
    in_pattern = '*.tif'

    logging.debug('Reading Files...')
    in_files = list(Path(in_dir).glob(in_pattern))
    in_names = [file.stem for file in in_files]
    in_shpfile = list(Path(in_dir).glob('*.shp'))[0]

    logging.debug('Opening files in GDAL...')
    temp = list(zip(in_names,in_files))
    imgs = {name: gdal.Open(str(file)) for name, file in tqdm(temp, position=0, leave=False)}
    shp = ogr.Open(str(in_shpfile))
    lyr = shp.GetLayer(0)
    

    logging.debug('Entering Data...')
    with open(in_dir+'.csv', 'w+') as f:
        f.write('point_id,x,y,geom,farm,sampling style,depth min,depth max,TC,IC,OC')
        [f.write(f',{k}')for k in imgs.keys()]
        f.write('\n')
        for ft in tqdm(lyr, position=0, leave=True):
            geom = ft.GetGeometryRef()
            point = geom.ExportToWkt()
            x, y, _ = geom.GetPoint()
            farm = ft.GetField(0)
            sampling_style = ft.GetField(1)
            point_id = ft.GetField(2)
            depth_range = ft.GetField(3)
            depth_min, depth_max = map(int, depth_range.split("-"))
            total_depth = ft.GetField(4)
            total_carbon = ft.GetField(5)
            inorganic_carbon = ft.GetField(6)
            organic_carbon = ft.GetField(7)
            f.write(f'{point_id},{x},{y},{point},{farm},{sampling_style},{depth_min},{depth_max},{total_carbon},{inorganic_carbon},{organic_carbon}')
            for k, v in tqdm(imgs.items(), position=1, leave=False):
                geot = v.GetGeoTransform()
                bnds = [v.GetRasterBand(i) for i in range(1, v.RasterCount+1)]
                for bnd in tqdm(bnds, position=1, leave=False):
                    col = int((x - geot[0]) / geot[1])
                    row = int((y - geot[3]) / geot[-1])

                    val = bnd.ReadAsArray(col, row, win_xsize=1, win_ysize=1)[0][0]
                    f.write(f',{val}')
            f.write('\n')

