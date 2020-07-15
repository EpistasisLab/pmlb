# -*- coding: utf-8 -*-

"""
PMLB was primarily developed at the University of Pennsylvania by:
    - Randal S. Olson (rso@randalolson.com)
    - William La Cava (lacava@upenn.edu)
    - Weixuan Fu (weixuanf@upenn.edu)
    - Trang Le (ttle@pennmedicine.upenn.edu)
    - and many more generous open source contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from .dataset_lists import (classification_dataset_names,
                            regression_dataset_names)

datasetmd = open("docs_sources/dataset.md", "w")
datasetmd.write("#Dataset List\n")
datasetmd.write("##Classification Benchmarks\n")
for d in classification_dataset_names:

    glink = ("https://github.com/EpistasisLab/"
            "penn-ml-benchmarks/blob/PMLB2.0/datasets")
    dlink = "- [{0}]({1}/{0}/README.md)\n".format(d, glink)
    datasetmd.write(dlink)

datasetmd.write("##Regression Benchmarks\n")
for d in regression_dataset_names:

    glink = ("https://github.com/EpistasisLab/"
            "penn-ml-benchmarks/blob/PMLB2.0/datasets")
    dlink = "- [{0}]({1}/{0}/README.md)\n".format(d, glink)
    datasetmd.write(dlink)
datasetmd.close()
