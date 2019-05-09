#!/usr/bin/env python

import os
import re
import numpy as np
import pandas as pd
import PyPDF2


def split_multipage_pdfs(directory, pdfname):
	"""
	This function splits a multipage pdf into its
	constituent pages. 

	Args:
		directory - the full path to the directory
			where the pdf file to be split is saved.
		pdfname - the name of the pdf file to be split
	Returns:
		pdfname_page*_.pdf - each of the individual
			pages of the pdf written to the same
			directory as the original pdf.

	"""

	filepath = '/'.join((directory, pdfname))
	inputpdf = PyPDF2.PdfFileReader(filepath, 'rb')

	for i in range(inputpdf.numPages):
		output = PyPDF2.PdfFileWriter()
		output.addPage(inputpdf.getPage(i))
		filename = '_'.join((filepath.split('.')[0], 'page', str(i)))
		with open(filename+'.pdf', 'wb') as outputStream:
			output.write(outputStream)