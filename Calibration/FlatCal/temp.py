	def __setupPlots(self):
		'''
		Initialize plotting variables
		'''
		flatCalPath,flatCalBasename = os.path.split(self.flatCalFileName)
		nPlotsPerRow = 3
		nPlotsPerCol = 4
		nPlotsPerPage = nPlotsPerRow*nPlotsPerCol
		iPlot = 0 
		pdfFullPath = self.calSolnPath+plotName+str(indexplot+1)+'.pdf'

	if os.path.isfile(pdfFullPath):
		answer = self.__query("{0} already exists. Overwrite?".format(pdfFullPath),
                                  yes_or_no=True)
		if answer is False:
			answer = self.__query("Provide a new file name (type exit to quit):")
			if answer == 'exit':
   				raise UserError("User doesn't want to overwrite the plot file " +
                                    "... exiting")
		pdfFullPath = self.calSolnPath+answer+str(indexplot+1)+'.pdf'
		while os.path.isfile(pdfFullPath):
			question = "{0} already exists. Choose a new file name " + \
                               "(type exit to quit):"
			answer = self.__query(question.format(pdfFullPath))
			if answer == 'exit':
				raise UserError("User doesn't want to overwrite the plot file " +
                                        "... exiting")
			pdfFullPath = self.calSolnPath+answer+str(indexplot+1)+'.pdf'
	else:
		os.remove(pdfFullPath)
		pp = PdfPages(pdfFullPath)

	def __mergePlots(self):
		'''
		Merge recently created temp.pdf with the main file
		'''
		pdfFullPath = self.calSolnPath+plotName+str(indexplot+1)+'.pdf'
		temp_file = os.path.join(self.calSolnPath, 'temp.pdf')
		if os.path.isfile(pdfFullPath):
			merger = PdfFileMerger()
			merger.append(PdfFileReader(open(plot_file, 'rb')))
			merger.append(PdfFileReader(open(temp_file, 'rb')))
			merger.write(plot_file)
			merger.close()
			os.remove(temp_file)
		else:
 			os.rename(temp_file, pdfFullPath)

	def __closePlots(self):
		'''
		Safely close plotting variables after plotting since the last page is only saved if it is full.
		'''
		if not self.saved:
			pdf = PdfPages(os.path.join(self.out_directory, 'temp.pdf'))
			pdf.savefig(self.fig)
 			pdf.close()
			self.__mergePlots()
		plt.close('all')
