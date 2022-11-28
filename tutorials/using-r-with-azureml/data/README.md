The data files here are two dataframes included in the 
[`bayesm`](https://cran.r-project.org/web/packages/bayesm/index.html) package. The files were loaded into an R environment and saved as two csv files.

* The `yx` dataframe was saved as `sales-data.csv`
* The `storedemo` dataframe was saved as `store-demographics.csv`


```r
library(bayesm)
data("orangeJuice")
names(orangeJuice)
write.csv(orangeJuice$yx, file = "sales-data.csv", na = "", row.names = F)
write.csv(orangeJuice$storedemo, file = "store-demographics.csv", na = "", row.names = F)
```
