# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # GitHub Miner

import os, shutil, sys, time, re, requests, csv, datetime, pytz
import pandas as pd
import numpy as np
import netCDF4 as nc
import xarray as xr
import warnings
import yaml
from github import Github
warnings.filterwarnings("ignore")

username = 'reedan88'
token = '03c5a5f29623dc3ae4439e547b409d95de6cb2b2'

g = Github(username, token)

repo = g.get_repo("ooi-integration/asset-management")
repo

pulls = repo.get_pulls(state='all', sort='merged', base='master')
for pr in pulls:
    if pr.merged:
        print(str(pr.number) + ': ' + pr.merged_at.strftime('%Y-%m-%d'))

# +
pulls = repo.get_pulls(state='all', sort='merged', base='master')

prNum = ()
prDate = ()
prFiles = ()

for pr in pulls:
    if pr.merged:
        if pr.merged_at > pd.to_datetime('2018-10-01'):
            prNum = prNum + (str(pr.number),)
            prDate = prDate + (pr.merged_at.strftime('%Y-%m-%d'),)
            files = []
            for file in pr.get_files():
                files.append(file.filename)
            prFiles = prFiles + (files,)
# -

df = pd.DataFrame(data=zip(prNum, prDate, prFiles), columns=['Pull Request #', 'Merge Date', 'Files'])

df

df.sort_values(by='Pull Request #', ascending=False, inplace = True)

df['Files'] = df['Files'].apply(lambda x: [y.split('/')[-1] for y in x])

lst = ['CE','AT','RS']

df['Files'] = df['Files'].apply(lambda x: [y for y in x if y[:2] not in lst])

df

df['Num of Files'] = df['Files'].apply(lambda x: len(x))

df

np.sum(df['Num of Files'])

df.to_csv('ooi_integration_pull_requests.csv')

files = []
for file in pr.get_files():
    files.append(file.filename)

files

for file in pr.get_files():
    print(file.filename)

file.sha

file.changes

file.deletions

file.contents_url

pr.

repo.get(sha=file.sha)


