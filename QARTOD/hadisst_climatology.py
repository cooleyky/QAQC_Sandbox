#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 19:44:28 2017

author: Craig Risien (crisien@coas.oregonstate.edu)
"""

import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
#from mpl_toolkits.basemap import Basemap

ds = xr.open_dataset('http://wilson.coas.oregonstate.edu:8080/thredds/dodsC/CIOSS/HadISST/Monthly/HadISST_sst.nc')
ds

for var in ds.variables:
    print(var)

sst = ds['sst'].values
lat = ds['latitude'].values
lon = ds['longitude'].values
time = ds['time'].values



sst = ds.sst.data
lat = ds.latitude.data
lon = ds.longitude.data
time = ds.time.data

del ds  #clear ds
#subset the data for the period 1900-1999
sst = sst[360:1560,:,:]
sst[sst==-1000]=np.nan
#reshape the sst matrix
tmp = np.zeros((180,360,1200))
for i in range(0, 1200):
        tmp[:,:,i] = sst[i,:,:]

sst=tmp
del tmp

time = time[360:1560]
#plot january 1900
lon, lat = np.meshgrid(lon,lat)
m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,\
            llcrnrlon=-180,urcrnrlon=180,resolution='c')
fig = plt.figure()
m.drawcoastlines(linewidth=0.25)
m.drawcountries(linewidth=0.25)
m.fillcontinents(color='grey',lake_color='aqua')
m.drawmapboundary(fill_color='white')
m.drawmeridians(np.arange(0,360,30),linewidth=0.25)
m.drawparallels(np.arange(-90,90,30),linewidth=0.25)
cax = m.pcolormesh(lon,lat,sst[:,:,1176], vmin=0, vmax=30, cmap='rainbow')
#plt.title('HadISST %s'%time[1176].astype('string'), fontsize=10)
cbar = fig.colorbar(cax, ticks=[0, 10, 20, 30], fraction=0.046, pad=0.04, shrink=.6)

#get sst array dims
[x,y,z]=np.shape(sst)
#these are monthly mean data so f=1/12
f=1./12.

beta_values = np.zeros((x,y,9))

for i in range(0, x):
    for j in range(0, y):
        c = np.squeeze(sst[i,j,:])
        t = np.arange(0, z)
        N=len(c)
        
        if np.sum(np.isnan(c)) >= 1:
            beta_values[i,j,:] = np.nan
        else:
            X=[np.ones(N),np.sin(2*np.pi*f*t),np.cos(2*np.pi*f*t),np.sin(4*np.pi*f*t),np.cos(4*np.pi*f*t),np.sin(6*np.pi*f*t),np.cos(6*np.pi*f*t),np.sin(8*np.pi*f*t),np.cos(8*np.pi*f*t)]
            [b,residues,rank,s] = np.linalg.lstsq(np.transpose(X),c,rcond=-1)
            beta_values[i,j,:] = b

#plot the mean field, i.e. beta_values[:,:,0]
m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,\
            llcrnrlon=-180,urcrnrlon=180,resolution='c')
fig = plt.figure()
m.drawcoastlines(linewidth=0.25)
m.drawcountries(linewidth=0.25)
m.fillcontinents(color='grey',lake_color='aqua')
m.drawmapboundary(fill_color='white')
m.drawmeridians(np.arange(0,360,30),linewidth=0.25)
m.drawparallels(np.arange(-90,90,30),linewidth=0.25)
cax = m.pcolormesh(lon,lat,beta_values[:,:,0], vmin=0, vmax=30, cmap='rainbow')
plt.title('Mean', fontsize=10)
cbar = fig.colorbar(cax, ticks=[0, 10, 20, 30], fraction=0.046, pad=0.04, shrink=.6)

new_t=np.arange(0, z)
seasonal_cycle = np.zeros((180,360,1200))

#using f=1/12 with calculate a monthly seasonal cycle, using 1/365 will calculate a daily seasonal cycle
for i in range(0, x):
    for j in range(0, y):
        T=(beta_values[i,j,0]+beta_values[i,j,1]*np.sin(2.*np.pi*f*new_t)+beta_values[i,j,2]*np.cos(2.*np.pi*f*new_t)+beta_values[i,j,3]*np.sin(4.*np.pi*f*new_t)+beta_values[i,j,4]*np.cos(4.*np.pi*f*new_t))
        seasonal_cycle[i,j,:] = T

sst_anomalies = sst-seasonal_cycle

#plot an example anomaly field
m = Basemap(projection='cyl',llcrnrlat=-90,urcrnrlat=90,\
            llcrnrlon=-180,urcrnrlon=180,resolution='c')
fig = plt.figure()
m.drawcoastlines(linewidth=0.25)
m.drawcountries(linewidth=0.25)
m.fillcontinents(color='grey',lake_color='aqua')
m.drawmapboundary(fill_color='white')
m.drawmeridians(np.arange(0,360,30),linewidth=0.25)
m.drawparallels(np.arange(-90,90,30),linewidth=0.25)
cax = m.pcolormesh(lon,lat,sst_anomalies[:,:,1176], vmin=-4, vmax=4, cmap='coolwarm')
#plt.title('HadISST Anomalies %s'%time[1176].astype('string'), fontsize=10)
cbar = fig.colorbar(cax, ticks=[-4, -2, 0, 2, 4], fraction=0.046, pad=0.04, shrink=.6)

#plot example time series
fig = plt.figure()
plt.plot(time[0:100],sst[130,125,0:100])
plt.plot(time[0:100],seasonal_cycle[130,125,0:100])
plt.legend(["HadISST", "Seasonal Cycle"],loc=1)
plt.ylabel('SST (oC)', fontsize=10)
#plt.title('%s'%lat[130,125].astype('string')+'N '+ '%s'%lon[130,125].astype('string')+'W', fontsize=10)
plt.show()

fig = plt.figure()
plt.plot(time,sst[130,125,:])
plt.plot(time,seasonal_cycle[130,125,:])
plt.legend(["HadISST", "Seasonal Cycle"],loc=1)
plt.ylabel('SST (oC)', fontsize=10)
#plt.title('%s'%lat[130,125].astype('string')+'N '+ '%s'%lon[130,125].astype('string')+'W', fontsize=10)
plt.show()
