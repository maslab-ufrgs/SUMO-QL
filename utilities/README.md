# plotme
 A script written in python that plots data from a file or from a dataframe, outputs that graph as an image or pdf. 

## Command Line Arguments

Description and valid values for each argument. Examples are presented in a different section.

### File Handling
| Verbose            | Short    | Default       | Description                                           | Valid Values                                           |
|--------------------|:--------:|:-------------:|:-----------------------------------------------------:|:------------------------------------------------------:|
| _--fileName_       | _-f_     | `required`    | Name of the files that contain the data for the graph. It can be a directory as well, as long as there are csv files in it. | `filename with or without path`.`extension` / `directory name with or without path`                 |
| _--fileExtension_       | _-ext_    | '.csv'          | File extension to be chosen if a directory is passed.             | `string`                        |
| _--dontSave_       | _-ds_     | Saves the plot    | Defines if plot will be saved. | - |
| _--displayPlot_       | _-dp_     | Doesn't display the plot    | Defines if plot will be displayed. | - |
| _--separator_ | _-sep_ | ,(comma) | Defines the separator used in the input file, for parsing purposes. | ' ', '\\t', regular expressions and other file delimiters |
| _--comment_           | _-com_    | #         |  The character that will indicate if a line should be treated as comment. | `string` |
| _--output_         | _-o_     | .pdf          | Name and/or extension of the output file.              | '.png', 'name', 'name.png'                             |

### Plot Configuration
| Verbose            | Short    | Default       | Description                                           | Valid Values                                           |
|--------------------|:--------:|:-------------:|:-----------------------------------------------------:|:------------------------------------------------------:|
| _--graphType_      | _-g_     | line          | Type of graph that will be plotted.                    | line, scatter, pie and bar                             |
| _--figSize_        | _-fig_   | auto        | Size of the graph and the exported image (Bounding Box). | `float,float`                                          |
| _--plotTitle_      | _-pt_    | none          | Title that appears at the top of the plot.             | `string`                                               |
| _--fontSize_       | _-fs_    | auto          | Size of the font used in the graph itself.             | `int`                                                  |
| _--showSpine_      | _-st_    | Doesn't show          | Shows the spines from the graph.                     | -                                     |
| _--symbols_        | _-s_     | point symbol  | Shape of the symbols used.     | Lists [ex: vhD] or values in https://matplotlib.org/3.1.0/api/markers_api.html |
| _--distBetSymbols_ | _-d_     | auto          | Distance between each symbol.                          | `int`                                                  |
| _--symbolSize_     | _-ss_    | auto          | Size of each symbol.                                   | `float`                                                |'   ** |
| _--lineWidth_      | _-l_     | auto          | Size of the line on a Line plot.                       | `int` or `float`                                       |
| _--pieLabel_       | _-pl_    | none          | Labels of the data in the pie plot.                           | `string1,string2,...,stringN`                          |


### Axis Configuration
| Verbose            | Short    | Default       | Description                                           | Valid Values                                           |
|--------------------|:--------:|:-------------:|:-----------------------------------------------------:|:------------------------------------------------------:|
| _--x_              | _-x_     | first column  | The x-axis of the plot.                                | Indexes of columns                             *       |
| _--y_              | _-y_     | second column | The y-axis of the plot.             | Indexes of columns(value, list [ex: 2,3,4] or sequences [ex: 2-4] *       |
| _--xmax_            |   -xmax   | auto          | Maximum value of x to be plotted.            | `float`                                                  |
| _--xmin_            |   -xmin   | auto          | Minimum value of x to be plotted.            | `float`                                                  |
| _--ymax_            |   -ymax   | auto          | Maximum value of y to be plotted.            | `float`                                                  |
| _--ymin_            |   -ymin   | auto          | Minimum value of y to be plotted.            | `float`                                                  |
| _--xLabel_         | _-xl_    | column name   | Label of the x-axis.                                   | `string`                                               |
| _--yLabel_         | _-yl_    | column name   | Label of the y-axis.                                   | `string`                                               |


### Color Configuration
| Verbose            | Short    | Default       | Description                                           | Valid Values                                           |
|--------------------|:--------:|:-------------:|:-----------------------------------------------------:|:------------------------------------------------------:|
| _--setPalette_     | _-p_     | colorblind    | Graph color palette.                                   | deep, muted, pastel, bright, dark and colorblind       |
| _--bgColor_        | _-bgc_   | lightgrey     | Changes the color of the background.                   | 'red','black','lightyellow','#abc','#ff701E'   **      |
| _--gColor_         | _-gc_    | grey          | Changes the color of the grid.                                     | 'red','black','lightyellow','#abc','#ff701E'   **      |
| _--colors_ | _-c_ | cycles default colors | Selects the colors of the plotted y-axes in the scatter and line plots. | 'red','black','lightyellow','#abc','#ff701E


### Miscellaneous
| Verbose            | Short    | Default       | Description                                           | Valid Values                                           |
|--------------------|:--------:|:-------------:|:-----------------------------------------------------:|:------------------------------------------------------:|
| _--movingAverageWindow_  | _-w_ | 1       | Plots the moving average, given the window | - |
| _--standardDeviation_  | _-sd_ | Doesn't calculate       | Makes a plot of the mean and the standard deviation over all the files, ploting the shadow. To plot standard deviation, there must be at least two files. If a directory is provided, it must only contain the files that are to be plotted. All files must have the same number of rows. | - |
| _--areaUnderCurve_  | _-auc_  | Doesn't calculate         | Calculates the area under the curve for a given file and the y index(es). If activated, doesn't generate a plot. Only accepts one file (if more than one files are given, will only calculate auc for the first file and ignore the others). | -                   |
| _--areaUnderCurveMethod_  | _-aucm_  | 'simpson'       | The method that is going to be used for the auc calculation. It can be the simpson rule, the trapezoidal rule or the mean of them. | 'simpson', 'trapz' or 'mean' |


* Column indexes begin at 1, not 0
** See https://matplotlib.org/stable/tutorials/colors/colors.html for more examples
*** 'lightblue', 'yellow', 'grey', 'lightpink', 'brown', 'pink', 'orange', 'green', 'dark yellow', 'blue'

## Examples
 - Line plot: `py plotme.py -f (path)filename.extension -x columnIndex -y columnIndex`
    - The only required argument is the filename 
 - Scatter plot with differente output name and plot title: `py plotme.py -f (path)file.ext -g scatter -pt title -o export` 
 - Bar plot: `py plotme.py -f (path)filename.extension`
    - The first item in the column is interpreted as the label of the axis, the subsequent itens in that column **NEED** to be of type int or float
 - Pie chart with labels on each slice while using a tab separated input file: `py plotme.py -f (path)filename.extension -sep '\t' -pl label,label2,...,labelN`
    - The first item in the column is interpreted as the label of the axis, the subsequent itens in that column **NEED** to be of type int or float
    - Each label corresponds to a single slice in the chart, from 1 to N, every label is assigned a slice following the file order. The number of labels need to be the same as the number of elements in the column.
	
### File Handling
   - f:
   <br/>
   `python3 plotme.py -f 'name with spaces.csv'`
   
   - ext:
   <br/>
   `python3 plotme.py -f dir -y 3-5 -sd -ext txt`
   
   - sp:
   <br/>
   `python3 plotme.py -f dir -sp False`
   
   - dp:
   <br/>
   `python3 plotme.py -f dir -dp True`
   
   - separator:
   <br/>
   `python3 plotme.py -f file.txt -sep \t`
   <br/>
   `python3 plotme.py -f file.csv -sep ,`
   <br/>
   `python3 plotme.py -f file.csv -sep ' '`
	
   - com:
   <br/>
   `python3 plotme.py -f file -com @`
   
   - output:
   <br/>
   `python3 plotme.py -f file -o outputFile`
   <br/>
   `python3 plotme.py -f file -o .tiff`
   <br/>
   `python3 plotme.py -f file -o export.jpeg`


### Plot Configuration
   - graphType:
   <br/>
   `python3 plotme.py -f file -g bar`
   
   - figSize:
   <br/>
   `python3 plotme.py -f file -fig 192,108`
   
   - plotTitle:
   <br/>
   `python3 plotme.py -f file -pt 'Tile of the plot'`
   
   - fontSize:
   <br/>
   `python3 plotme.py -f file -fs 14`
   
   - hideSpine:
   <br/>
   `python3 plotme.py -f file -st True`
   
   - symbols:
   <br/>
   `python3 plotme.py -f file -y 2,3 -s vH`
   <br/>
   `python3 plotme.py -f file -y 2-6 -s vHDdv`
   <br/>
   `python3 plotme.py -f file -y 2-6 -s v`
   
   - distBetSymbols:
   <br/>
   `python3 plotme.py -f file -d 3`
   
   - symbolSize:
   <br/>
   `python3 plotme.py -f file -s v -ss 15`
   
   - lineWidth:
   <br/>
   `python3 plotme.py -f file -l 15`
   
   - pieLabel:
   <br/>
   `python3 plotme.py -f file -g pie -pl slice1,'another slice',3`


### Axis Configuration
   - x:
   <br/>
   `python3 plotme.py -f file -x 1`            
   
   - y:
   <br/>
   `python3 plotme.py -f file -y 5-7`
   <br/>
   `python3 plotme.py -f file -y 5,6,7`
   
   - xmax:
   <br/>
   `python3 plotme.py -f file -xmax 10`
   
   - xmin:
   <br/>
   `python3 plotme.py -f file -xmin 10`
   
   - ymax:
   <br/>
   `python3 plotme.py -f file -ymax 10`
   
   - ymin:
   <br/>
   `python3 plotme.py -f file -ymin 10`
   
   - xLabel:
   <br/>
   `python3 plotme.py -f file -xl 'label of x'`
   
   - yLabel:
   <br/>
   `python3 plotme.py -f file -yl 'label of y'`


### Color Configuration
   - setPalette:
   <br/>
   `python3 plotme.py -f file -p deep`
   
   - bgColor:
   <br/>
   `python3 plotme.py -f file -bgc black`
   <br/>
   `python3 plotme.py -f file -bgc '#ff701E'`
   
   - gColor:
   <br/>
   `python3 plotme.py -f file '#abc'`
   
   - colors:
   <br/>
   `python3 plotme.py -f file -y 2,4 -c yellow,green`
   <br/>
   `python3 plotme.py -f file -y 2,4 -c '#ff701E','#abc'`


### Miscellaneous
   - w:
   <br/>
   `python3 plotme.py -f file1 -w 100`
   <br/>
   `python3 plotme.py -f file1 file2 file3 [...] -y 2 -sd -w 1000`
   
   - sd:
   <br/>
   `python3 plotme.py -f file1 file2 file3 [...] -y 2 -sd`
   <br/>
   `python3 plotme.py -f file1 file2 file3 -y 2-4,7 -sd`
   <br/>
   `python3 plotme.py -f directory -y 2 -sd` 
   
   - auc:
   <br/>
   `python3 plotme.py -f file -y 4-6 -auc`
   <br/>
   `python3 plotme.py -f file1 file2 file3 [...] -y 3,4 -auc`
   
   - aucm:
   <br/>
   `python3 plotme.py -f file -y 4-6 -auc -aucm simpson`
   <br/>
   `python3 plotme.py -f file1 file2 file3 [...] -y 3,4 -auc -aucm trapz`


## Using it as an imported module

 1. After importing, you need to make an instance of the `Plot` class while passing, at least, the  `data`(the imported version of fileName) argument with the dataframe, the rest of the arguments have the same names as their CLI counterparts. 
 2. Call the `plotGraph()` method. The file will be exported as `Plot.pdf` if no `output` argument was passed.

### Use python3, as well as pip3 to install the dependencies

### Dependencies: seaborn, matplotlib, pandas, argparse, re, ast, itertools
