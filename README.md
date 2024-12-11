# CPSC545_CourseProject
Course Project on Traditional Clustering Methods on Skin Cancer Cell Datasets

The cell data and plot information associated with 4 papers are included separately in the folders labeled paper#_data.

To run the python scripts, it may be necessary to alter the file path to the data. Each paper and set of data has its own python script since the data is formatted and thus handled differently between papers. The python scripts labeled Paper9_UMAP.py can be modified to run both the SCC and BCC datasets. Modify the lines 89-96 as indicated in the comments to run either option. 

Each script has a separate function for k-means, DBSCAN and agglomerative clustering. Once all the clusering labels are returned, the main function uses sklearn to calculate the different assessment metrics. The metrics were collected in google sheets for computing the graphical presentation of the project results. These graphs can be found here: [https://docs.google.com/spreadsheets/d/1LphStUgn2Z7bzH9Wl0uwoQm7P_LTeFiDuIaqvOnlSHY/edit?usp=sharing]
