from math import *

import pandas as pd
import numpy as np
import datetime


def geodistance(lng1, lat1, lng2, lat2):
    lng1, lat1, lng2, lat2 = map(radians, [float(lng1), float(lat1), float(lng2), float(lat2)])  # 经纬度转换成弧度
    dlon = lng2 - lng1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    distance = 2 * asin(sqrt(a)) * 6371 * 1000  # 地球平均半径，6371km
    distance = round(distance / 1000, 3)

    return distance


LAT_PER_METER = 8.993203677616966e-06
LNG_PER_METER = 1.1700193970443768e-05
basic_info = {}
# 最大纬度40.19988,最小纬度39.5018,最大经度116.79979,最小经度116.00211000000002
basic_info['beijing'] = [39.5018, 116.00211, 137]
# 最大纬度22.99977,最小纬度22.46103,最大经度114.59611,最小经度113.61387
basic_info['shenzhen'] = [22.46103, 113.61387, 168]
# 最大纬度23.066811,最小纬度22.78582,最大经度43.72261,最小经度43.160049
basic_info['baxi'] = [22.78582, 43.160049, 97]
# 最大纬度31.032499,最小纬度30.290665,最大经度104.609691,最小经度103.269706
basic_info['chengdu'] = [30.290665, 103.269706, 230]

def getRowCol(grid, type='beijing'):
    row = (grid) // basic_info[type][2]
    col = (grid) % basic_info[type][2]
    return row, col


# 计算两个网格之间的距离
def calDisByGrid(grid1, grid2, type):
    if grid1 == grid2:
        return 0
    row1, col1 = getRowCol(grid1)
    row2, col2 = getRowCol(grid2)

    lat1 = basic_info[type][0] + row1 * 500 * LAT_PER_METER
    lat2 = basic_info[type][0] + row2 * 500 * LAT_PER_METER
    lng1 = basic_info[type][1] + col1 * 500 * LNG_PER_METER
    lng2 = basic_info[type][1] + col2 * 500 * LNG_PER_METER

    return geodistance(lng1, lat1, lng2, lat2)


# 计算修复网格和真实点之间的距离
def calDisBylatlng(true_grid, pred_grid, true_lat, true_lng, type):
    row1, col1 = getRowCol(pred_grid, type)
    pred_lat = basic_info[type][0] + row1 * 500 * LAT_PER_METER + 250 * LAT_PER_METER
    pred_lng = basic_info[type][1] + col1 * 500 * LNG_PER_METER + 250 * LNG_PER_METER
    # print(true_grid, pred_grid, true_lat, true_lng, pred_lat, pred_lng,
    #       geodistance(pred_lng, pred_lat, true_lng, true_lat))
    # exit()
    return geodistance(pred_lng, pred_lat, true_lng, true_lat)


# 通过经纬度计算网格序列号
def calGridBylatlng(lat, lng, type='beijing'):
    lat_unit = LAT_PER_METER * 500
    lng_unit = LNG_PER_METER * 500
    locgrid_x = int((lng - basic_info[type][1]) // lng_unit)
    locgrid_y = int((lat - basic_info[type][0]) // lat_unit)

    return locgrid_y * basic_info[type][2] + locgrid_x + 2

# print(calDisBylatlng(1, 4773, 39.659088, 116.701278, 'beijing'))
# print(calGridBylatlng(39.731231, 116.48972))





























































































