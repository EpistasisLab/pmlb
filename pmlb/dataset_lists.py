# -*- coding: utf-8 -*-

"""
PMLB was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - William La Cava (lacava@upenn.edu)
    - Weixuan Fu (weixuanf@upenn.edu)
    - and many more generous open source contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
import pathlib, pandas
from io import StringIO
import pkgutil
data = pkgutil.get_data('pmlb', 'all_summary_stats.tsv')
df_summary = pandas.read_csv(StringIO(data.decode("utf-8")) , sep='\t')
regression_dataset_names = df_summary.query('task=="regression"')['dataset'].tolist()
classification_dataset_names = df_summary.query('task=="classification"')['dataset'].tolist()
dataset_names = regression_dataset_names + classification_dataset_names

# reviewed_datasets = [
#     'molecular_biology_promoters',
#     'car',
#     'connect_4',
#     'dna',
#     '542_pollution',
#     '560_bodyfat',
#     'poker',
#     '1089_USCrime',
#     '529_pollen',
#     'chess',
#     'penguins',
#     'bupa',
#     'movement_libras',
#     'adult',
#     'waveform_21',
#     'waveform_40',
#     'saheart',
#     'wine_quality_white',
#     'wine_quality_red',
#     'irish',
#     'mushroom'
# ]
