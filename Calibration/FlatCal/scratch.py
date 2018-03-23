	def writeWeights(self):
		"""
		Writes an h5 file to put calculated flat cal factors in
		"""
		for index, flatWeightfile in enumerate(self.flatWeightsList):
			if os.path.isabs(self.flatCalFileName) == True:
				fullFlatCalFileName = self.flatCalFileName+str(index+1)
			else:
				scratchDir = os.getenv('MKID_PROC_PATH')
				flatDir = os.path.join(scratchDir,'flatCalSolnFiles')
				fullFlatCalFileName = os.path.join(flatDir,self.flatCalFileName+str(index+1))

			if not os.path.exists(fullFlatCalFileName) and self.calSolnPath =='':	
				os.makedirs(fullFlatCalFileName)		

			try:
				flatCalFile = tables.open_file(fullFlatCalFileName,mode='w')
			except:
				print('Error: Couldn\'t create flat cal file, ',fullFlatCalFileName)
				return
			print('wrote to',self.flatCalFileName)

			calgroup = flatCalFile.create_group(flatCalFile.root,'flatcal','Table of flat calibration weights by pixel and wavelength')
			calarray = tables.Array(calgroup,'weights',obj=self.alltheflatWeights[index,:,:,:],title='Flat calibration Weights indexed by pixelRow,pixelCol,wavelengthBin')
			flagtable = tables.Array(calgroup,'flags',obj=self.flatFlags,title='Flat cal flags indexed by pixelRow,pixelCol,wavelengthBin. 0 is Good')
			bintable = tables.Array(calgroup,'wavelengthBins',obj=self.wvlBinEdges,title='Wavelength bin edges corresponding to third dimension of weights array')

			descriptionDict = FlatCalSoln_Description(self.nWvlBins)
			caltable = flatCalFile.create_table(calgroup, 'calsoln', descriptionDict,title='Flat Cal Table')
        
			for iRow in range(self.nXPix):
				for iCol in range(self.nYPix):
					weights = self.flatWeights[iRow,iCol,:]
					deltaWeights = self.deltaFlatWeights[iRow,iCol,:]
					flags = self.flatFlags[iRow,iCol,:]
					flag = np.any(self.flatFlags[iRow,iCol,:])
					pixelName = self.beamImage[iRow,iCol]

					entry = caltable.row
					entry['resid'] = pixelName
					entry['pixelrow'] = iRow
					entry['pixelcol'] = iCol
					entry['weights'] = weights
					entry['weightUncertainties'] = deltaWeights
					entry['weightFlags'] = flags
					entry['flag'] = flag
					entry.append()
        
			flatCalFile.flush()
			flatCalFile.close()
	
			npzFileName = os.path.splitext(fullFlatCalFileName)[0]+'.npz'
