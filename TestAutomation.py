import pandas as pd
import lazypredict
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyRegressor


data1 = pd.read_csv(r"D:\Capstone Data\F01.csv")
data2 = pd.read_csv(r"D:\Capstone Data\F02.csv")
data3 = pd.read_csv(r"D:\Capstone Data\F03.csv")

d1Px = data1.drop(['TC','IC','OC','LS8_F01_SR_B1', 'LS8_F01_SR_B2',
                   'LS8_F01_SR_B3', 'LS8_F01_SR_B4', 'LS8_F01_SR_B5', 
                   'LS8_F01_SR_B6','LS8_F01_SR_B7', 'S2_F01_B1', 
                   'S2_F01_B11', 'S2_F01_B12', 'S2_F01_B2','S2_F01_B3', 
                   'S2_F01_B4', 'S2_F01_B5', 'S2_F01_B6', 'S2_F01_B7',
                   'S2_F01_B8', 'S2_F01_B8A', 'S2_F01_B9'], axis=1)
d1Sx = data1.drop(['TC','IC','OC','X01_F01', 'X02_F01', 'X03_F01',
                   'X04_F01', 'X05_F01', 'X06_F01', 'X07_F01', 'X08_F01', 
                   'X09_F01','X10_F01', 'X11_F01', 'X12_F01', 'X13_F01', 
                   'X14_F01', 'X15_F01','X16_F01', 'X17_F01', 'X18_F01', 
                   'X19_F01'], axis=1)
d1Bx = data1.drop(['TC','IC','OC'], axis=1)
d2Px = data2.drop(['TC','IC','OC','LS8_F02_SR_B1', 'LS8_F02_SR_B2',
                   'LS8_F02_SR_B3', 'LS8_F02_SR_B4', 'LS8_F02_SR_B5', 
                   'LS8_F02_SR_B6','LS8_F02_SR_B7', 'S2_F02_B1', 
                   'S2_F02_B11', 'S2_F02_B12', 'S2_F02_B2','S2_F02_B3', 
                   'S2_F02_B4', 'S2_F02_B5', 'S2_F02_B6', 'S2_F02_B7',
                   'S2_F02_B8', 'S2_F02_B8A', 'S2_F02_B9'], axis=1)
d2Sx = data2.drop(['TC','IC','OC','X01_F02', 'X02_F02', 'X03_F02',
                   'X04_F02', 'X05_F02', 'X06_F02', 'X07_F02', 'X08_F02', 
                   'X09_F02','X10_F02', 'X11_F02', 'X12_F02', 'X13_F02', 
                   'X14_F02', 'X15_F02','X16_F02', 'X17_F02', 'X18_F02', 
                   'X19_F02'], axis=1)
d2Bx = data2.drop(['TC','IC','OC'], axis=1)
d3Px = data3.drop(['TC','IC','OC','LS8_F03_SR_B1', 'LS8_F03_SR_B2',
                   'LS8_F03_SR_B3', 'LS8_F03_SR_B4', 'LS8_F03_SR_B5', 
                   'LS8_F03_SR_B6','LS8_F03_SR_B7', 'S2_F03_B1', 
                   'S2_F03_B11', 'S2_F03_B12', 'S2_F03_B2','S2_F03_B3', 
                   'S2_F03_B4', 'S2_F03_B5', 'S2_F03_B6', 'S2_F03_B7',
                   'S2_F03_B8', 'S2_F03_B8A', 'S2_F03_B9'], axis=1)
d3Sx = data3.drop(['TC','IC','OC','X01_F03', 'X02_F03', 'X03_F03',
                   'X04_F03', 'X05_F03', 'X06_F03', 'X07_F03', 'X08_F03', 
                   'X09_F03','X10_F03', 'X11_F03', 'X12_F03', 'X13_F03', 
                   'X14_F03', 'X15_F03','X16_F03', 'X17_F03', 'X18_F03', 
                   'X19_F03'], axis=1)
d3Bx = data3.drop(['TC','IC','OC'], axis=1)

xss = [[d1Px,d1Sx,d1Bx],[d2Px,d2Sx,d2Bx],[d3Px,d3Sx,d3Bx]]

d1Ty = data1['TC']
d1Iy = data1['IC']
d1Oy = data1['OC']

d2Ty = data2['TC']
d2Iy = data2['IC']
d2Oy = data2['OC']

d3Ty = data3['TC']
d3Iy = data3['IC']
d3Oy = data3['OC']

yss = [[d1Ty,d1Iy,d1Oy],[d2Ty,d2Iy,d2Oy],[d3Ty,d3Iy,d3Oy]]

reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)

with open(r"D:\Capstone Data\models30p.txt", 'w+') as f:
    for i, (fxs, fys) in enumerate(zip(xss,yss), start=1):
        f.write(f'Farm {i}:\n')
        for j, x in enumerate(fxs):
            for k, y in enumerate(fys):
                f.write(f'{j}, {k}:\n')
                x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3, random_state = 64)
                reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
                models,pred = reg.fit(x_train, x_test, y_train, y_test)
                f.write(f'{models}\n')

